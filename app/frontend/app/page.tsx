"use client";

import { useState, useRef, useCallback } from "react";

const CONDITIONS = [
  { value: "clean", label: "Clean", icon: "✦" },
  { value: "noise", label: "Noise", icon: "⬡" },
  { value: "blur", label: "Blur", icon: "◎" },
  { value: "low_light", label: "Low Light", icon: "◑" },
  { value: "jpeg", label: "JPEG Artifact", icon: "▦" },
] as const;

type ConditionValue = (typeof CONDITIONS)[number]["value"];

type ModelResult = {
  model: string;
  prediction: string;
  confidence: number;
  latency_ms: number;
};

type CompareResponse = {
  condition: string;
  results: ModelResult[];
};

const MODEL_LABELS: Record<string, string> = {
  baseline_cnn: "Baseline CNN",
  lightweight_cnn: "Lightweight CNN",
  eagle_net: "EAGLE-Net",
};

const MODEL_COLORS: Record<string, string> = {
  baseline_cnn: "#4a9eff",
  lightweight_cnn: "#a78bfa",
  eagle_net: "#34d399",
};

// ── Model type badges ────────────────────────────────────────────────────────
const MODEL_BADGES: Record<string, { label: string; style: React.CSSProperties }> = {
  baseline_cnn: {
    label: "Standard",
    style: {
      background: "rgba(74,158,255,0.12)",
      border: "1px solid rgba(74,158,255,0.35)",
      color: "#4a9eff",
    },
  },
  lightweight_cnn: {
    label: "Fast",
    style: {
      background: "rgba(167,139,250,0.12)",
      border: "1px solid rgba(167,139,250,0.35)",
      color: "#a78bfa",
    },
  },
  eagle_net: {
    label: "Robust",
    style: {
      background: "rgba(52,211,153,0.12)",
      border: "1px solid rgba(52,211,153,0.35)",
      color: "#34d399",
    },
  },
};

// ── CSS filter per distortion condition ─────────────────────────────────────
const CONDITION_FILTERS: Record<ConditionValue, string> = {
  clean: "none",
  blur: "blur(3.5px)",
  low_light: "brightness(0.30) contrast(1.1)",
  noise: "contrast(1.08) saturate(0.88)",
  jpeg: "contrast(1.06) saturate(1.08)",
};

// ── Condition descriptions for insight text ──────────────────────────────────
const CONDITION_DESCRIPTIONS: Record<string, string> = {
  clean: "clean, undistorted",
  noise: "noise-corrupted",
  blur: "motion-blurred",
  low_light: "low-light",
  jpeg: "JPEG-compressed",
};

// ── Insight sentence builder ─────────────────────────────────────────────────
function buildInsight(topModel: ModelResult, condition: string): string {
  const name = MODEL_LABELS[topModel.model] ?? topModel.model;
  const conf = (topModel.confidence * 100).toFixed(1);
  const condDesc = CONDITION_DESCRIPTIONS[condition] ?? condition;

  const notes: Record<string, string> = {
    baseline_cnn: "suggesting the standard architecture handles this distortion adequately.",
    lightweight_cnn: "demonstrating that its compact design sacrifices little accuracy here.",
    eagle_net: "indicating stronger robustness and superior feature extraction under distortion.",
  };
  const note = notes[topModel.model] ?? "outperforming the other models on this input.";

  return `${name} achieves the highest confidence (${conf}%) under ${condDesc} conditions, ${note}`;
}

// ── Confidence bar ───────────────────────────────────────────────────────────
function ConfidenceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono w-12 text-right" style={{ color }}>
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ── Scan line ────────────────────────────────────────────────────────────────
function ScanLine() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none rounded-2xl">
      <div
        className="absolute left-0 right-0 h-px opacity-20"
        style={{
          background: "linear-gradient(90deg, transparent, #34d399, transparent)",
          animation: "scanline 3s linear infinite",
        }}
      />
    </div>
  );
}

// ── Dual image preview ───────────────────────────────────────────────────────
function DualImagePreview({
  preview,
  condition,
  fileName,
}: {
  preview: string;
  condition: ConditionValue;
  fileName?: string;
}) {
  const condLabel = CONDITIONS.find((c) => c.value === condition)?.label ?? condition;

  return (
    <div
      className="card p-5 relative overflow-hidden"
      style={{ animation: "fade-up 0.35s ease forwards" }}
    >
      <ScanLine />
      <p className="mono text-xs mb-4" style={{ color: "#34d399", letterSpacing: "0.1em" }}>
        ◈ IMAGE PREVIEW
      </p>

      <div className="grid grid-cols-2 gap-4">
        {/* Original */}
        <div>
          <p className="mono text-xs mb-2" style={{ color: "#4a6580", letterSpacing: "0.08em" }}>
            ORIGINAL
          </p>
          <div
            className="relative rounded-xl overflow-hidden"
            style={{ border: "1px solid rgba(74,158,255,0.25)" }}
          >
            <img
              src={preview}
              alt="Original"
              className="w-full h-44 object-cover block"
            />
            <span
              className="absolute bottom-2 left-2 tag"
              style={{
                background: "rgba(0,0,0,0.65)",
                border: "1px solid rgba(74,158,255,0.3)",
                color: "#4a9eff",
              }}
            >
              SOURCE
            </span>
          </div>
        </div>

        {/* Transformed */}
        <div>
          <p className="mono text-xs mb-2" style={{ color: "#4a6580", letterSpacing: "0.08em" }}>
            TRANSFORMED · {condLabel.toUpperCase()}
          </p>
          <div
            className="relative rounded-xl overflow-hidden"
            style={{ border: "1px solid rgba(52,211,153,0.25)" }}
          >
            <img
              src={preview}
              alt={`${condLabel} transformed`}
              className="w-full h-44 object-cover block"
              style={{ filter: CONDITION_FILTERS[condition] }}
            />

            {/* Grain overlay for noise */}
            {condition === "noise" && (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.2'/%3E%3C/svg%3E")`,
                  backgroundSize: "160px 160px",
                  mixBlendMode: "overlay",
                  opacity: 0.6,
                }}
              />
            )}

            {/* Banding lines for JPEG artifact */}
            {condition === "jpeg" && (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  backgroundImage:
                    "repeating-linear-gradient(0deg, transparent, transparent 7px, rgba(0,0,0,0.07) 8px)",
                  mixBlendMode: "multiply",
                }}
              />
            )}

            <span
              className="absolute bottom-2 left-2 tag"
              style={{
                background: "rgba(0,0,0,0.65)",
                border: "1px solid rgba(52,211,153,0.3)",
                color: "#34d399",
              }}
            >
              {condLabel.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {fileName && (
        <p className="mono text-xs mt-3" style={{ color: "#4a6580" }}>
          ↳ {fileName}
        </p>
      )}
    </div>
  );
}

// ── Inference loading card ───────────────────────────────────────────────────
function InferenceLoader() {
  return (
    <div
      className="card p-8 flex flex-col items-center gap-5"
      style={{
        border: "1px solid rgba(52,211,153,0.25)",
        animation: "fade-up 0.25s ease forwards",
      }}
    >
      {/* Radar ring */}
      <div className="relative flex items-center justify-center" style={{ width: 64, height: 64 }}>
        <div
          className="absolute rounded-full"
          style={{
            inset: 0,
            border: "1.5px solid rgba(52,211,153,0.2)",
            animation: "radar-expand 1.6s ease-out infinite",
          }}
        />
        <div
          className="absolute rounded-full"
          style={{
            inset: 8,
            border: "1.5px solid rgba(52,211,153,0.3)",
            animation: "radar-expand 1.6s ease-out 0.4s infinite",
          }}
        />
        <div
          className="rounded-full"
          style={{
            width: 16,
            height: 16,
            background: "rgba(52,211,153,0.85)",
            boxShadow: "0 0 14px #34d399",
          }}
        />
      </div>

      <div className="text-center">
        <p className="mono text-sm font-bold" style={{ color: "#34d399", letterSpacing: "0.12em" }}>
          RUNNING INFERENCE...
        </p>
        <p className="mono text-xs mt-1" style={{ color: "#4a6580" }}>
          Classifying across all models
        </p>
      </div>

      {/* Bouncing dots */}
      <div className="flex gap-2">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="rounded-full"
            style={{
              width: 6,
              height: 6,
              background: "#34d399",
              animation: `dot-bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

// ── Insight panel ────────────────────────────────────────────────────────────
function InsightPanel({
  topModel,
  condition,
}: {
  topModel: ModelResult;
  condition: string;
}) {
  const color = MODEL_COLORS[topModel.model] ?? "#8fa3b8";
  const badge = MODEL_BADGES[topModel.model];

  return (
    <div
      className="card p-5 relative overflow-hidden"
      style={{
        border: `1px solid ${color}33`,
        background: `${color}08`,
        animation: "fade-up 0.4s ease 0.2s forwards",
        opacity: 0,
      }}
    >
      <div className="flex items-start gap-4">
        {/* Icon bubble */}
        <div
          className="flex-shrink-0 flex items-center justify-center rounded-xl"
          style={{
            width: 42,
            height: 42,
            background: `${color}18`,
            border: `1px solid ${color}33`,
          }}
        >
          <span style={{ color, fontSize: "1.1rem" }}>◎</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <p className="mono text-xs" style={{ color, letterSpacing: "0.1em" }}>
              ◈ INSIGHT
            </p>
            {badge && (
              <span className="tag" style={badge.style}>
                {badge.label}
              </span>
            )}
          </div>
          <p className="text-sm leading-relaxed" style={{ color: "#c2d4e4" }}>
            {buildInsight(topModel, condition)}
          </p>
        </div>
      </div>

      {/* Stat strip */}
      <div
        className="grid grid-cols-3 gap-3 mt-4 pt-4"
        style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}
      >
        <div className="text-center">
          <p className="mono text-xs mb-1" style={{ color: "#4a6580" }}>TOP CONF.</p>
          <p className="mono text-sm font-bold" style={{ color }}>
            {(topModel.confidence * 100).toFixed(1)}%
          </p>
        </div>
        <div className="text-center">
          <p className="mono text-xs mb-1" style={{ color: "#4a6580" }}>LATENCY</p>
          <p className="mono text-sm font-bold" style={{ color }}>
            {topModel.latency_ms.toFixed(1)} ms
          </p>
        </div>
        <div className="text-center">
          <p className="mono text-xs mb-1" style={{ color: "#4a6580" }}>CONDITION</p>
          <p className="mono text-sm font-bold" style={{ color: "#4a9eff" }}>
            {condition.toUpperCase()}
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function EagleNetPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [condition, setCondition] = useState<ConditionValue>("clean");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResults(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);

    const selectedCondition: ConditionValue = condition;
    const form = new FormData();
    form.append("file", file);
    // UI labels like "Low Light" differ from backend values, and the current
    // FastAPI route reads condition from the query string. Send the selected
    // backend value in both places so it does not fall back to "clean".
    form.set("condition", selectedCondition);
    const compareUrl = new URL("http://localhost:8000/compare");
    compareUrl.searchParams.set("condition", selectedCondition);

    try {
      const res = await fetch(compareUrl.toString(), {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: CompareResponse = await res.json();
      setResults(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  const topModel = results?.results.reduce((a, b) =>
    a.confidence > b.confidence ? a : b
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

        * { box-sizing: border-box; }

        body {
          background: #050a0f;
          color: #e2eaf2;
          font-family: 'Syne', sans-serif;
          min-height: 100vh;
          margin: 0;
        }

        @keyframes scanline {
          0% { top: -1px; }
          100% { top: 101%; }
        }

        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 0.6; }
          100% { transform: scale(1.6); opacity: 0; }
        }

        @keyframes fade-up {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes shimmer {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }

        @keyframes radar-expand {
          0%   { transform: scale(0.5); opacity: 0.8; }
          100% { transform: scale(2.2); opacity: 0; }
        }

        @keyframes dot-bounce {
          0%, 80%, 100% { transform: translateY(0);   opacity: 0.3; }
          40%            { transform: translateY(-7px); opacity: 1;   }
        }

        .grid-bg {
          background-image:
            linear-gradient(rgba(52,211,153,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(52,211,153,0.04) 1px, transparent 1px);
          background-size: 40px 40px;
        }

        .card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          backdrop-filter: blur(12px);
        }

        .card-accent {
          background: rgba(52,211,153,0.04);
          border: 1px solid rgba(52,211,153,0.2);
          border-radius: 16px;
        }

        .btn-primary {
          background: linear-gradient(135deg, #34d399, #059669);
          color: #000;
          font-family: 'Space Mono', monospace;
          font-weight: 700;
          font-size: 0.8rem;
          letter-spacing: 0.1em;
          border: none;
          border-radius: 10px;
          padding: 14px 32px;
          cursor: pointer;
          transition: all 0.2s;
          position: relative;
          overflow: hidden;
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 8px 24px rgba(52,211,153,0.35);
        }

        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .upload-zone {
          border: 1.5px dashed rgba(52,211,153,0.3);
          border-radius: 14px;
          transition: all 0.2s;
          cursor: pointer;
        }

        .upload-zone:hover, .upload-zone.dragging {
          border-color: rgba(52,211,153,0.7);
          background: rgba(52,211,153,0.04);
        }

        .select-custom {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 10px;
          color: #e2eaf2;
          font-family: 'Space Mono', monospace;
          font-size: 0.82rem;
          padding: 10px 14px;
          appearance: none;
          cursor: pointer;
          transition: border-color 0.2s;
          width: 100%;
        }

        .select-custom:focus {
          outline: none;
          border-color: rgba(52,211,153,0.5);
        }

        /* Result rows: fade-up stagger + hover lift */
        .result-row {
          animation: fade-up 0.4s ease forwards;
          transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .result-row:nth-child(1) { animation-delay: 0.04s; opacity: 0; }
        .result-row:nth-child(2) { animation-delay: 0.12s; opacity: 0; }
        .result-row:nth-child(3) { animation-delay: 0.20s; opacity: 0; }

        .result-row:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 22px rgba(0,0,0,0.4);
        }

        .mono { font-family: 'Space Mono', monospace; }

        .tag {
          font-family: 'Space Mono', monospace;
          font-size: 0.62rem;
          letter-spacing: 0.12em;
          padding: 2px 8px;
          border-radius: 4px;
          text-transform: uppercase;
        }

        .eagle-badge {
          background: rgba(52,211,153,0.15);
          border: 1px solid rgba(52,211,153,0.4);
          color: #34d399;
        }

        .dot-pulse {
          width: 8px; height: 8px;
          border-radius: 50%;
          background: #34d399;
          position: relative;
          display: inline-block;
        }

        .dot-pulse::after {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: 50%;
          background: #34d399;
          animation: pulse-ring 1.2s ease-out infinite;
        }

        .shimmer-text {
          background: linear-gradient(90deg, #34d399, #a78bfa, #4a9eff, #34d399);
          background-size: 300% auto;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          animation: shimmer 4s linear infinite;
        }
      `}</style>

      <div className="grid-bg min-h-screen py-12 px-4">
        {/* Header */}
        <div className="max-w-4xl mx-auto mb-10 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="dot-pulse" />
            <span className="tag" style={{ background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.3)", color: "#34d399" }}>
              LIVE INFERENCE
            </span>
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight mb-3" style={{ letterSpacing: "-0.02em" }}>
            <span className="shimmer-text">EAGLE-Net</span>
            <br />
            <span style={{ color: "#8fa3b8", fontSize: "0.5em", fontWeight: 600, letterSpacing: "0.25em", textTransform: "uppercase" }}>
              Model Comparison
            </span>
          </h1>
          <p className="mono text-xs" style={{ color: "#4a6580", letterSpacing: "0.05em" }}>
            CNN ensemble · satellite image classification · visual distortion analysis
          </p>
        </div>

        <div className="max-w-4xl mx-auto grid gap-5">
          {/* Upload + Config Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Upload */}
            <div className="card p-5 relative overflow-hidden">
              <ScanLine />
              <p className="mono text-xs mb-3" style={{ color: "#34d399", letterSpacing: "0.1em" }}>
                ◈ IMAGE INPUT
              </p>
              <div
                className={`upload-zone p-6 text-center ${dragging ? "dragging" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                {preview ? (
                  <div className="space-y-3">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full h-36 object-cover rounded-lg"
                      style={{ border: "1px solid rgba(52,211,153,0.2)" }}
                    />
                    <p className="mono text-xs" style={{ color: "#4a9eff" }}>
                      {file?.name}
                    </p>
                    <p className="text-xs" style={{ color: "#4a6580" }}>Click to replace</p>
                  </div>
                ) : (
                  <div className="py-6">
                    <div className="text-4xl mb-3 opacity-30">⬡</div>
                    <p className="text-sm font-semibold mb-1" style={{ color: "#8fa3b8" }}>
                      Drop image here
                    </p>
                    <p className="mono text-xs" style={{ color: "#4a6580" }}>
                      PNG · JPG · WEBP
                    </p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </div>

            {/* Config */}
            <div className="card p-5 flex flex-col justify-between">
              <div>
                <p className="mono text-xs mb-4" style={{ color: "#34d399", letterSpacing: "0.1em" }}>
                  ◈ DISTORTION CONDITION
                </p>
                <div className="space-y-2">
                  {CONDITIONS.map((c) => (
                    <label
                      key={c.value}
                      className="flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all"
                      style={{
                        background: condition === c.value ? "rgba(52,211,153,0.08)" : "transparent",
                        border: `1px solid ${condition === c.value ? "rgba(52,211,153,0.3)" : "transparent"}`,
                      }}
                    >
                      <input
                        type="radio"
                        name="condition"
                        value={c.value}
                        checked={condition === c.value}
                        onChange={() => setCondition(c.value)}
                        className="hidden"
                      />
                      <span className="text-lg" style={{ opacity: condition === c.value ? 1 : 0.3 }}>
                        {c.icon}
                      </span>
                      <span
                        className="text-sm font-semibold"
                        style={{ color: condition === c.value ? "#34d399" : "#8fa3b8" }}
                      >
                        {c.label}
                      </span>
                      {condition === c.value && (
                        <span className="ml-auto tag eagle-badge">selected</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Dual image preview — shown as soon as a file is loaded */}
          {preview && (
            <DualImagePreview
              preview={preview}
              condition={condition}
              fileName={file?.name}
            />
          )}

          {/* Submit */}
          <div className="flex justify-center">
            <button
              className="btn-primary"
              onClick={handleSubmit}
              disabled={!file || loading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.3" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  RUNNING INFERENCE...
                </span>
              ) : (
                "▶  RUN COMPARISON"
              )}
            </button>
          </div>

          {/* Enhanced loading card */}
          {loading && <InferenceLoader />}

          {/* Error */}
          {error && (
            <div
              className="card p-4 mono text-sm text-center"
              style={{ borderColor: "rgba(248,113,113,0.3)", color: "#f87171" }}
            >
              ✕ &nbsp;{error}
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="card-accent p-6 relative overflow-hidden">
              <ScanLine />

              <div className="flex items-center justify-between mb-5">
                <p className="mono text-xs" style={{ color: "#34d399", letterSpacing: "0.1em" }}>
                  ◈ INFERENCE RESULTS
                </p>
                <div className="flex items-center gap-2">
                  <span className="tag" style={{ background: "rgba(74,158,255,0.1)", border: "1px solid rgba(74,158,255,0.3)", color: "#4a9eff" }}>
                    {results.condition.toUpperCase()}
                  </span>
                  {topModel && (
                    <span className="tag eagle-badge">
                      Best: {MODEL_LABELS[topModel.model] ?? topModel.model}
                    </span>
                  )}
                </div>
              </div>

              {/* Table Header */}
              <div
                className="grid grid-cols-4 mono text-xs mb-2 px-4"
                style={{ color: "#4a6580", letterSpacing: "0.08em" }}
              >
                <span>MODEL</span>
                <span>PREDICTION</span>
                <span className="col-span-2">CONFIDENCE · LATENCY</span>
              </div>

              <div className="space-y-3">
                {[...results.results]
                  .sort((a, b) => b.confidence - a.confidence)
                  .map((r, i) => {
                    const color = MODEL_COLORS[r.model] ?? "#8fa3b8";
                    const isTop = r.model === topModel?.model;
                    const badge = MODEL_BADGES[r.model];
                    return (
                      <div
                        key={r.model}
                        className="result-row grid grid-cols-4 items-center gap-4 p-4 rounded-2xl"
                        style={{
                          background: isTop ? "rgba(52,211,153,0.06)" : "rgba(255,255,255,0.02)",
                          border: `1px solid ${isTop ? "rgba(52,211,153,0.2)" : "rgba(255,255,255,0.05)"}`,
                          animationDelay: `${i * 0.08}s`,
                        }}
                      >
                        {/* Model name + type badge */}
                        <div>
                          <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                            {isTop && (
                              <span style={{ color: "#fbbf24", fontSize: "0.75rem" }}>★</span>
                            )}
                            <span className="text-xs font-bold mono" style={{ color }}>
                              {MODEL_LABELS[r.model] ?? r.model}
                            </span>
                          </div>
                          {/* Model type badge */}
                          {badge && (
                            <span className="tag" style={{ ...badge.style, fontSize: "0.58rem" }}>
                              {badge.label}
                            </span>
                          )}
                        </div>

                        {/* Prediction */}
                        <div>
                          <span className="text-sm font-bold" style={{ color: "#e2eaf2" }}>
                            {r.prediction}
                          </span>
                        </div>

                        {/* Confidence bar + latency */}
                        <div className="col-span-2 space-y-1">
                          <ConfidenceBar value={r.confidence} color={color} />
                          <div className="mono text-xs" style={{ color: "#4a6580" }}>
                            ⏱ {r.latency_ms.toFixed(1)} ms
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>

              {/* Summary strip */}
              {topModel && (
                <div
                  className="mt-5 p-3 rounded-xl flex items-center justify-between"
                  style={{ background: "rgba(52,211,153,0.05)", border: "1px solid rgba(52,211,153,0.15)" }}
                >
                  <span className="mono text-xs" style={{ color: "#4a6580" }}>
                    Top performer under{" "}
                    <span style={{ color: "#4a9eff" }}>{results.condition}</span> condition
                  </span>
                  <span className="mono text-xs font-bold" style={{ color: "#34d399" }}>
                    {MODEL_LABELS[topModel.model]} · {(topModel.confidence * 100).toFixed(1)}% conf
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Insight panel — rendered below results */}
          {results && topModel && (
            <InsightPanel topModel={topModel} condition={results.condition} />
          )}
        </div>

        {/* Footer */}
        <p className="text-center mono text-xs mt-12" style={{ color: "#1e3040" }}>
          EAGLE-NET © 2026 · CNN SATELLITE CLASSIFICATION BENCHMARK
        </p>
      </div>
    </>
  );
}