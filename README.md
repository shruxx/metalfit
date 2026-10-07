# metalfit

Runs GGUF models on an Apple Silicon Mac the way that Mac actually wants them: it reads a model's headers,
works out whether it fits the GPU **whole**, says what would make it fit if it does not, starts `llama-server`
with those settings, and keeps one address in front of the model so the thing you chat with does not need
reconfiguring when you switch.

It is not a chat interface. Point [Open WebUI](https://github.com/open-webui/open-webui), an editor, or anything
else that speaks the OpenAI API at `http://127.0.0.1:8099/v1` and you have one. What metalfit does is the part
nothing else does: the arithmetic, and the switching.

```
$ metalfit list
Kolibri-1-Q3_K_M.gguf
  architecture   kolibri1, 50 blocks (-ngl 51 is all of it)
  on disk        37.5 GB
  in memory      34.9 GiB of weights
  KV cache       0.68 GiB at -c 65536 (10 layers context-sized, 40 capped at 513)
  Metal          37.44 GiB working set
  -> all 51 layers on the GPU at -c 65536, 0.6 GiB spare

Kolibri-1-Q4_K_M.gguf
  in memory      44.2 GiB of weights
  -> does not fit whole at any context: the weights alone need 44.2 GiB of 37.4 GiB, 8.0 GiB too much
```

## Why "does it fit whole" is the only question

On Apple Silicon the CPU and the GPU share one memory, so the usual advice - put as many layers on the GPU as
will go - is wrong in a way that costs more than half the speed. What matters is a threshold: once nearly every
tensor in the file belongs to the GPU's buffer, llama.cpp **maps** the file into it instead of **copying** the
GPU's share out of it. Mapped, the pages stay file-backed, nothing competes for dirty memory, and nothing is read
from the SSD while the model answers. One layer short of that line, every GPU layer costs a copy and takes memory
away from the page cache the CPU-side weights are read through.

Measured on a MacBook Pro M5 Pro, 48 GB, macOS 27.0.1, llama.cpp build 782, Qwen3.8-Flash-Next Q2_0 at `-c 32768`,
through a server with varied prompts:

| GPU layers | tok/s | SSD per token | footprint | load |
|---:|---:|---:|---:|---:|
| 44 | 23.8 | 1.2 - 8.5 MB | 32.2 GiB | 31.7 s |
| 47 | 29.2 | 0.2 MB | 0.9 GiB | 11.8 s |
| 49 (all) | 33.4 | 0.2 MB | 0.7 GiB | |

The footprint falling by 31 GiB and the load getting 20 s shorter between 44 and 47 layers *is* the copy
disappearing. Kolibri-1 went the same way: 32 tok/s as a split at Q4_K_M, 59 whole at Q3_K_M.

So metalfit does not try to pick a clever split. It works out whether the model fits, and if it does it passes
`-ngl <all>`; if it does not, it says so and leaves the split to llama.cpp's own `--fit`, because choosing one
well needs measurement on the machine, not arithmetic.

## What it computes, and how it is checked

- **What has to be in memory.** Not the file size: llama.cpp leaves a tensor in the file and reads its rows on
  demand when the architecture marks it and it is over 4 GiB (`LLAMA_LAZY_MODE_AUTO`). Only `qwen4exp` and
  `gemma4` mark anything, both their per-layer embedding table - 28.8 GB of Qwen3.8-Flash-Next's 66.4, which
  needs no memory at all.
- **The KV cache, per architecture.** A flat allowance gets this wrong by a factor of four. Kolibri-1 caches
  only 513 tokens on 40 of its 50 layers, so 128K context costs it 1.35 GiB instead of ~7 GB. Qwen3.8-Flash-Next
  is hybrid: 12 of 48 layers have a KV cache and 36 keep a recurrent state that does not grow with the context.
  metalfit reads `attention.sliding_window_pattern`, `attention.compress_ratios` and the `ssm.*` keys to tell
  these apart. Qwen3.5 / 3.6 (`qwen35`, `qwen35moe`) give only `full_attention_interval`; counted as every
  layer, Qwen3.8-27B needed 17.27 GiB of KV at 128K instead of 4.66. gemma4 gives its KV heads per layer (2 on
  the full layers, 8 on the sliding ones), which read as one number came out as no KV cache at all.
- **Tensor sizes without knowing the quantization.** From the gaps between data offsets, never from the type,
  because the interesting files are the ones with types too new for the tooling. `GGML_TYPE_Q2_0` is 42 of 43,
  and neither the `gguf` PyPI package nor Ollama's importer can size a tensor that uses it - Ollama refuses such
  a file with `unsupported tensor ... size overflows`.
- **The compute reserve** (1.2 GiB) is derived from two measurements rather than guessed: on the stock 37.44 GiB
  working set Qwen Q2_0 leaves 2.44 GiB beside its weights, and `-c 65536` (0.80 GiB of KV) runs while
  `-c 131072` (1.60 GiB) dies with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. That brackets everything else
  between 0.84 and 1.64 GiB.

`tests/test_fit.py` holds those measurements as assertions. If a change to the arithmetic stops predicting what
the Mac did, the change is wrong until the Mac says otherwise.

## Install and use

**The app, with everything in it.** On a Mac with Apple Silicon and macOS 14 or newer:

```bash
curl -fsSL https://github.com/shruxx/metalfit/releases/latest/download/install.sh | sh
```

That puts `MetalfitBar.app` into `~/Applications` and starts it: a menu bar item that runs the server, shows
the loaded model and switches between models. The app carries its own Python 3.12, metalfit and a Metal
`llama-server` (31 MB zipped), so nothing else needs installing; the script checks the download against the
release's `SHA256SUMS`. Put `.gguf` files into `~/models` - they appear in the menu within 2 s - and point a chat
app at `http://127.0.0.1:8099/v1`. Running the same line again updates it. `release/build.sh` makes the
release; the app is signed ad hoc, not with a Developer ID, so a zip downloaded in a browser instead of with
the script has to be opened once with right-click > Open.

**From the source.** Python 3.11 or newer, no dependencies, and a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp) built with Metal.

```bash
pip install -e .

metalfit fit model.gguf                 # what one model needs
metalfit list --models ~/models         # every model in a folder, with its verdict
metalfit serve --models ~/models        # the page and the API on 127.0.0.1:8099
```

`metalfit serve` opens a small page to start and stop models, and passes `/v1/*` through to whichever one is
running - streamed replies included. Switching is one llama-server stopping and another starting, which for a
35 GiB model that maps takes about 13 seconds.

**It can leave the Mac usable while the model runs.** What the GPU holds is wired and macOS cannot page it out,
so a model held whole pushes browsers and editors into swap; the layers left off the GPU are mapped file pages it
simply drops under pressure. `--keep-free-gib N` puts on only as many layers as leaves N GB alone. Kolibri-1
Q3_K_M (34.9 GiB) on a 48 GB M5 Pro, `-c 65536`, measured through the server:

| GPU layers | wired | free for everything else | tok/s |
|---:|---:|---:|---:|
| 51 (all) | 40.2 GiB | 7.8 GB | 62.0 |
| 36 | 29.5 GiB | 18.5 GB | 39.1 |
| 26 (`--keep-free-gib 24`) | 22.5 GiB | 25.5 GB | 33.8 |
| 16 | 15.5 GiB | 32.5 GB | 24.3 |

**It gives the memory back when you stop using it.** A model on the GPU whole wires nearly all of it - 39.8 GiB
of a 48 GB Mac for a 35 GiB model - and wired memory cannot be paged out, so while it sits there unused every
browser tab and editor on the machine goes to swap instead. After ten idle minutes (`--keep-alive`, 0 to switch
it off) metalfit unloads, and the next request loads it again, which took 17.8 s in the same measurement. A
request naming a model loads that model, so a chat app can switch without touching this page, and `/v1/models`
answers from the folder whether or not anything is running.

**In the menu bar.** `menubar/build.sh` builds `MetalfitBar.app` (Swift, AppKit only, needs `swiftc`): the
loaded model's name in the menu bar, and a menu to switch models, unload, and start or stop the server. Its
mark is a chip filled by a stack of layers, and the fill is the state: every layer when a model is loaded, half
while one loads, none when the server is up with nothing loaded, a dashed outline when the server is off. The
app icon is the same mark, drawn by the same code (`menubar/Icon.swift`). It
only uses the API above, polled every 2 s, so it also follows a server started by hand. Stopping sends
SIGTERM to whatever listens on the port, which metalfit answers by unloading, so no llama-server stays behind
with its memory wired. With "Open at login" and "Start server when this app opens" in its menu, the server
comes up at every login. An open menu is rebuilt on every poll, so a model deleted from the folder is gone
from it within 2 s. Keep metalfit and llama-server out of `~/Documents`, `~/Desktop` and `~/Downloads`
(`uv tool install .` puts metalfit in `~/.local/bin`): macOS asks the app for access to those folders, and
until someone answers, the server hangs in its first `open()`.

`--llama-server <path>` if it is not next to you or on `PATH`; `METALFIT_LLAMA_SERVER` does the same.
`--working-set-gib` overrides what Metal reports, which is useful for asking "what would fit if I raised
`iogpu.wired_limit_mb`" without raising it.

## What it does not do

- **No chat interface, no model downloads, no registry.** Open WebUI chats; Hugging Face has the models.
- **One model at a time.** Two large models do not fit a shared memory; pretending otherwise would just page.
- **It does not pick a split** for a model that does not fit - see above.
- **Apple Silicon only.** The whole point is Metal's working set and the unified memory, neither of which a
  discrete GPU has.
- **Tested on two Macs**: an M5 Pro with 48 GB against two model families, and a MacBook Pro M1 Max with
  32 GB (27.0 GiB working set) running Qwen3.6-35B-A3B whole at `-c 32768`. There, through the server at
  temperature 0, three ~500-token replies averaged 50.2 tok/s at UD-Q2_K_XL, 48.1 at UD-Q3_K_XL, 44.5 at
  UD-Q4_K_XL and UD-IQ4_XS, and 41.1 at MXFP4_MOE; after an 8969-token prompt all of them wrote 28-31 tok/s.
  Numbers from other Macs would be welcome.

## Acknowledgements

The measurements behind the arithmetic were made while working on
[StrataForMac](https://github.com/shruxx/StrataForMac) (MIT), whose `tools/strata_runner.py` first worked out
that the lazy table must not be counted and that the layer count has to be measured rather than assumed. The
copy-versus-map behaviour is llama.cpp's, with Strata's `metal-split-mmap.patch` doing the copying below the
threshold.

MIT licensed.
