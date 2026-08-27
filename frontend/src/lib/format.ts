export function fmtBytes(n?: number | null): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(1) + " " + u[i];
}

export function fmtNum(n?: number): string {
  if (!n) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function kindBadge(kind: string): string {
  switch (kind) {
    case "llm": return "success";
    case "onnx": return "default";
    case "paddle": return "warning";
    default: return "secondary";
  }
}