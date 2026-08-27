// Declaration shim for the Wails-generated JS bindings (no .d.ts is emitted
// by wails3 generate bindings). The real JS provides these exports.
declare module "@bindings/cpm_orc" {
  export const RuntimeService: any;
  export const HFHubService: any;
  export const OcrService: any;
  export const LlmService: any;
  export const AsrService: any;
}