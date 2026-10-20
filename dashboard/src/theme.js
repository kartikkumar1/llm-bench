import { useEffect, useState } from "react";

// Color follows the backend, never its position in the current filter.
export const BACKENDS = [
  { key: "llamacpp_server", label: "llama.cpp C++ (GPU)", color: "--series-1" },
  { key: "llamacpp_python", label: "llama-cpp-python (CPU)", color: "--series-2" },
  { key: "hf_transformers", label: "HF Transformers", color: "--series-3" },
  { key: "llamacpp_server_cpu", label: "llama.cpp C++ (CPU)", color: "--series-4" },
];

const TOKENS = ["--surface-1", "--text-primary", "--text-secondary", "--text-muted", "--grid", "--axis",
  "--series-1", "--series-2", "--series-3", "--series-4", "--good", "--critical"];

function read() {
  const css = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((t) => [t, css.getPropertyValue(t).trim()]));
}

// Recharts writes colors as SVG attributes, so resolve the CSS tokens to hex and
// re-resolve when the OS theme or a data-theme stamp changes.
export function useTheme() {
  const [theme, setTheme] = useState(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(read());
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", update); obs.disconnect(); };
  }, []);
  return theme;
}
