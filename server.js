require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');

const app = express();

// ---------------------------------------------------------------------------
// Konfigurasi
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;
const LLM_BASE_URL = (process.env.LLM_BASE_URL || 'https://litellm.koboi2026.biz.id/v1').replace(/\/+$/, '');
const LLM_API_KEY = process.env.LLM_API_KEY || '';
const LLM_MODEL = process.env.LLM_MODEL || 'gemini/gemini-2.5-flash';
const MAX_IMAGE_DIMENSION = parseInt(process.env.MAX_IMAGE_DIMENSION || '768', 10);
const JPEG_QUALITY = parseInt(process.env.JPEG_QUALITY || '70', 10);

app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 }, // 8MB batas upload mentah
});

// ---------------------------------------------------------------------------
// GREEN COMPUTING — estimasi energi & karbon dari pipeline itu sendiri.
// Angka ini heuristik sederhana untuk memberi gambaran efisiensi, bukan
// pengukuran presisi laboratorium.
// ---------------------------------------------------------------------------
const ENERGY_PER_MB_KWH = 0.00035; // estimasi energi transfer+inferensi per MB payload
const GRID_CARBON_KG_PER_KWH = 0.45; // rata-rata faktor emisi grid (kg CO2e / kWh)

function estimateGreenMetrics(originalBytes, compressedBytes, elapsedMs) {
  const originalMB = originalBytes / (1024 * 1024);
  const compressedMB = compressedBytes / (1024 * 1024);
  const energySavedKWh = Math.max(0, (originalMB - compressedMB) * ENERGY_PER_MB_KWH);
  const co2SavedKg = energySavedKWh * GRID_CARBON_KG_PER_KWH;
  const reductionPercent = originalBytes > 0
    ? Math.round((1 - compressedBytes / originalBytes) * 100)
    : 0;

  return {
    ukuran_asli_kb: Math.round(originalBytes / 1024),
    ukuran_terkompresi_kb: Math.round(compressedBytes / 1024),
    reduksi_ukuran_persen: reductionPercent,
    estimasi_energi_dihemat_kwh: Number(energySavedKWh.toFixed(6)),
    estimasi_co2_dihemat_kg: Number(co2SavedKg.toFixed(6)),
    waktu_proses_ms: elapsedMs,
    strategi: 'Kompresi & resize gambar di sisi server sebelum dikirim ke LLM (edge preprocessing) untuk mengurangi payload, token, dan konsumsi energi inferensi.',
  };
}

// ---------------------------------------------------------------------------
// Preprocessing gambar (green computing: kecilkan payload sebelum ke LLM)
// ---------------------------------------------------------------------------
async function compressImage(buffer) {
  const compressed = await sharp(buffer)
    .rotate() // auto-orient berdasarkan EXIF
    .resize({
      width: MAX_IMAGE_DIMENSION,
      height: MAX_IMAGE_DIMENSION,
      fit: 'inside',
      withoutEnlargement: true,
    })
    .jpeg({ quality: JPEG_QUALITY, mozjpeg: true })
    .toBuffer();
  return compressed;
}

// ---------------------------------------------------------------------------
// Prompt sistem — dibatasi ketat ke limbah pisang & nangka saja supaya hemat
// token: kalau bukan itu, model wajib membalas objek singkat "valid: false"
// tanpa menjabarkan 12 poin analisis.
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `Kamu adalah sistem visi komputer untuk menganalisis LIMBAH ORGANIK dari PISANG (kulit, bonggol, jantung, tandan) dan NANGKA (kulit, dami, biji, sisa daging) saja.

ATURAN KETAT:
- Jika gambar BUKAN limbah pisang atau nangka (misalnya limbah lain, buah utuh non-limbah, benda lain, atau tidak jelas), balas HANYA JSON singkat: {"valid": false, "pesan": "<alasan singkat 1 kalimat, Bahasa Indonesia>"}. JANGAN isi field lain, JANGAN beri penjelasan tambahan.
- Jika gambar ADALAH limbah pisang atau nangka, balas HANYA JSON sesuai skema di bawah, semua nilai teks dalam Bahasa Indonesia, semua angka berupa number (bukan string), TANPA markdown, TANPA backtick, TANPA teks pembuka/penutup.

SKEMA JIKA VALID:
{
  "valid": true,
  "jenis_limbah": string,
  "tingkat_kepercayaan_persen": number,
  "jejak_karbon": { "nilai_kgco2e_per_kg": number, "penjelasan": string },
  "estimasi_berat_gram": number,
  "rekomendasi_pemanfaatan": [string, string, string],
  "kesegaran": { "tingkat": "segar" | "mulai_layu" | "agak_busuk" | "busuk", "skor_0_100": number, "deskripsi": string },
  "deteksi_kontaminasi": { "terdeteksi": boolean, "jenis_benda_asing": [string], "catatan": string },
  "optimasi_ekstraksi": { "suhu_celcius": number, "waktu_menit": number, "energi_kwh": number, "metode": string },
  "efisiensi_komputasi_model": { "rekomendasi": "edge" | "cloud", "alasan": string, "estimasi_hemat_energi_persen": number },
  "kadar_biokimia_estimasi": { "pektin_persen": number, "selulosa_persen": number, "pati_persen": number },
  "prediksi_shelflife": { "hari_tersisa": number, "catatan": string },
  "optimasi_rute_pengumpulan": { "rekomendasi": string, "estimasi_jarak_km": number, "moda_transport": string },
  "proyeksi_nilai_tambah_ekonomi": { "potensi_produk": string, "estimasi_nilai_rp_per_kg": number },
  "penghematan_biaya_tpa": { "estimasi_kg_terhindar_dari_tpa": number, "estimasi_hemat_rp": number, "catatan": string }
}

Semua angka adalah estimasi visual masuk akal berdasarkan kondisi gambar, bukan hasil laboratorium. Jangan pernah menambahkan teks di luar objek JSON.`;

// ---------------------------------------------------------------------------
// Helper: panggil LiteLLM (OpenAI-compatible) dengan gambar base64
// ---------------------------------------------------------------------------
async function callVisionLLM(base64Jpeg) {
  const res = await fetch(`${LLM_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${LLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      temperature: 0.2,
      max_tokens: 2500,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Analisis gambar limbah berikut sesuai instruksi sistem.' },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Jpeg}` } },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`LLM proxy error ${res.status}: ${errText.slice(0, 500)}`);
  }

  const data = await res.json();
  const raw = data?.choices?.[0]?.message?.content ?? '';
  return raw;
}

function parseModelJson(raw) {
  let text = String(raw).trim();
  // Buang pagar markdown kalau model tetap membungkusnya
  text = text.replace(/^```json/i, '').replace(/^```/, '').replace(/```$/, '').trim();

  // Ambil substring JSON pertama yang valid kalau ada teks nyasar di sekitarnya
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    text = text.slice(firstBrace, lastBrace + 1);
  }

  return JSON.parse(text);
}

// ---------------------------------------------------------------------------
// Route: analisis gambar
// ---------------------------------------------------------------------------
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  const startedAt = Date.now();

  try {
    if (!req.file || !req.file.buffer || req.file.buffer.length === 0) {
      return res.status(400).json({ ok: false, error: 'Tidak ada gambar yang dikirim.' });
    }

    const originalBytes = req.file.buffer.length;

    // Green computing: kompres & resize dulu sebelum kirim ke LLM
    const compressedBuffer = await compressImage(req.file.buffer);
    const base64Jpeg = compressedBuffer.toString('base64');

    const raw = await callVisionLLM(base64Jpeg);

    let parsed;
    try {
      parsed = parseModelJson(raw);
    } catch (e) {
      return res.status(502).json({
        ok: false,
        error: 'Gagal membaca respons model sebagai JSON.',
        raw_preview: String(raw).slice(0, 300),
      });
    }

    const elapsedMs = Date.now() - startedAt;
    const green = estimateGreenMetrics(originalBytes, compressedBuffer.length, elapsedMs);

    if (parsed && parsed.valid === false) {
      return res.status(200).json({
        ok: true,
        valid: false,
        pesan: parsed.pesan || 'Gambar tidak terdeteksi sebagai limbah pisang atau nangka.',
        green_computing: green,
      });
    }

    return res.status(200).json({
      ok: true,
      valid: true,
      hasil: parsed,
      green_computing: green,
    });
  } catch (err) {
    console.error('[analyze] error:', err.message);
    return res.status(500).json({ ok: false, error: 'Terjadi kesalahan pada server.', detail: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ ok: true, model: LLM_MODEL, base_url: LLM_BASE_URL });
});

app.listen(PORT, () => {
  console.log(`Waste Analyzer backend jalan di http://localhost:${PORT}`);
  console.log(`LLM: ${LLM_MODEL} @ ${LLM_BASE_URL}`);
});
