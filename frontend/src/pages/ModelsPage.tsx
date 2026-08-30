import { useState, useCallback, useEffect } from "react";
import { Search, Download, FolderOpen, FolderPlus, Trash2, ChevronDown, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import { fmtBytes, fmtNum, kindBadge } from "@/lib/format";
import { HFHubService, RuntimeService } from "@bindings/cpm_orc/internal/app";

export default function ModelsPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [locals, setLocals] = useState<any[]>([]);
  const [root, setRoot] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [files, setFiles] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [onnxOnly, setOnnxOnly] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setRoot(await HFHubService.ModelRoot());
      setLocals(await HFHubService.LocalModels());
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Debounced search as you type.
  useEffect(() => {
    const q = query.trim();
    if (!q) { setResults([]); setSearchError(null); setSearching(false); return; }
    setSearching(true);
    setSearchError(null);
    const t = setTimeout(async () => {
      try {
        let res = await HFHubService.SearchModel(q, 30);
        if (onnxOnly) res = res.filter((m: any) => m.tags?.some((x: string) => x.includes("onnx")));
        setResults(res);
      } catch (e: any) {
        setSearchError(friendlySearchError(String(e)));
      } finally {
        setSearching(false);
      }
    }, 450);
    return () => clearTimeout(t);
  }, [query, onnxOnly]);

const friendlySearchError = (raw: string): string => {
    if (/(timeout|dial tcp|EOF|connection|tls)/i.test(raw)) {
      return "无法连接 HuggingFace。请到「运行环境」页配置网络代理（如 http://127.0.0.1:7890）后重试。";
    }
    return "搜索失败: " + raw;
  };

  const doSearch = () => {
    // force immediate search
    setQuery((q) => q);
    if (!query.trim()) toast("请输入关键词", true);
  };

  const download = async (id: string) => {
    try {
      await HFHubService.DownloadModel(id, [], "");
      toast("下载完成: " + id);
      refresh();
    } catch (e) { toast("下载失败: " + e, true); }
  };

  const toggleFiles = async (id: string) => {
    if (expanded === id) { setExpanded(null); return; }
    setExpanded(id);
    try {
      const list = await HFHubService.ModelFiles(id, "", "");
      setFiles(list.filter((f: any) => f.type === "file"));
    } catch (e) { toast(String(e), true); setFiles([]); }
  };

  const del = async (id: string) => {
    try { await HFHubService.DeleteModel(id); refresh(); }
    catch (e) { toast(String(e), true); }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">模型管理</h1>
        <p className="mt-1 text-sm text-muted-foreground">搜索 HuggingFace Hub，下载并管理本地模型（ONNX / Paddle / LLM）。</p>
      </div>

      <Card>
        <CardContent className="space-y-3 pt-5">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input className="pl-9 pr-10" placeholder="搜索模型，如 MiniCPM / PP-OCR / Qwen3（输入即搜）"
                value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doSearch()} />
              {searching && <Loader2 className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-primary" />}
            </div>
            <Button onClick={doSearch} disabled={searching}><Search className="h-4 w-4" />搜索</Button>
            <label className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <Switch checked={onnxOnly} onCheckedChange={setOnnxOnly} />仅 ONNX
            </label>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FolderOpen className="h-4 w-4" /> 本地目录
            <Input className="h-8 flex-1" value={root} onChange={(e) => setRoot(e.target.value)} />
            <Button size="sm" variant="outline" onClick={async () => { try { await HFHubService.SetModelRoot(root); toast("已更新"); refresh(); } catch (e) { toast(String(e), true); } }}>设置</Button>
            <Button size="sm" variant="outline" onClick={() => RuntimeService.OpenFolder(root)}>打开</Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        {/* Remote */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><Search className="h-4 w-4" />远程搜索<Badge>{results.length}</Badge></CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {searchError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                {searchError}
                <Button size="sm" variant="outline" className="mt-2" onClick={() => doSearch()}>
                  <RefreshCw className="h-3.5 w-3.5" />重试
                </Button>
              </div>
            )}
            {!searchError && results.length === 0 && !searching && (
              <p className="text-sm text-muted-foreground">输入关键词搜索，回车或点击「搜索」</p>
            )}
            {searching && <p className="text-sm text-muted-foreground">搜索中…</p>}
            {results.map((m) => (
              <div key={m.id} className="rounded-lg border bg-muted/50 p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="break-all text-sm font-medium">{m.id}</div>
                    <div className="mt-0.5 text-xs text-muted-foreground">
                      ↓ {fmtNum(m.downloads)} · ♥ {fmtNum(m.likes)} · {m.pipelineTag || "model"}
                      {m.tags?.some((t: string) => t.includes("onnx")) && <Badge className="ml-2">onnx</Badge>}
                    </div>
                  </div>
                  <div className="flex shrink-0 gap-1.5">
                    <Button size="sm" onClick={() => download(m.id)}><Download className="h-3.5 w-3.5" />下载</Button>
                    <Button size="sm" variant="outline" onClick={() => toggleFiles(m.id)}><ChevronDown className="h-3.5 w-3.5" /></Button>
                  </div>
                </div>
                {expanded === m.id && (
                  <div className="mt-3 max-h-52 space-y-1 overflow-y-auto">
                    {files.map((f) => (
                      <div key={f.path} className="flex items-center justify-between rounded bg-muted px-2 py-1 text-xs">
                        <span className="font-mono text-[11px]">{f.path}</span>
                        <span className="text-muted-foreground">{fmtBytes(f.size || f.lfs?.size)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Local */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><FolderOpen className="h-4 w-4" />本地模型<Badge>{locals.length}</Badge></CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {locals.length === 0 && <p className="text-sm text-muted-foreground">本地还没有模型</p>}
            {locals.map((m) => (
              <div key={m.id} className="flex items-start justify-between gap-2 rounded-lg border bg-muted/50 p-3">
                <div className="min-w-0">
                  <div className="break-all text-sm font-medium">
                    {m.id}
                    <Badge className="ml-2" variant={kindBadge(m.kind) as any}>{m.kind}</Badge>
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground">{fmtBytes(m.size)} · {m.modified}</div>
                </div>
                <div className="flex shrink-0 gap-1.5">
                  <Button size="sm" variant="outline" onClick={() => HFHubService.OpenModel(m.id)}><FolderPlus className="h-3.5 w-3.5" /></Button>
                  <Button size="sm" variant="destructive" onClick={() => del(m.id)}><Trash2 className="h-3.5 w-3.5" /></Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}