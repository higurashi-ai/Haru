# Third-party notices

Haru is licensed under the Apache License 2.0 (see `LICENSE`). It includes or
derives from the following third-party work.

## VibeVoice acoustic tokenizer

The audio codec (`haru/haru/models/vibevoice_codec.py`,
`haru/haru/codec_triton/`, `haru_mlx/haru_mlx/models/vibevoice_codec.py`) implements
the architecture of the VibeVoice acoustic tokenizer, and the released model
weights include its parameters.

- Copyright (c) 2025 Microsoft
- License: MIT

## Moshi web client

The web client (`client/`, and its built bundle in `haru/haru/static/` and
`haru_mlx/haru_mlx/static/`) is derived from the Moshi web client.

- Copyright (c) Kyutai
- License: MIT (see `client/LICENSE`)

## mlx-lm

`haru_mlx/haru_mlx/models/gated_delta.py` and `haru_mlx/haru_mlx/models/qwen3_5.py`
are derived from the Qwen3.5 model code of mlx-lm.

- Copyright (c) Apple Inc.
- License: MIT

## Bundled in the web client build

| Package | License |
| --- | --- |
| opus-recorder (Chris Rudmin), including libopus and the speexdsp resampler (Xiph.Org Foundation and contributors) | MIT; BSD-3-Clause |
| react, react-dom, scheduler (Meta Platforms, Inc.) | MIT |
| lucide-react | ISC |
| webm-duration-fix | ISC |
| buffer, ebml-block, int64-buffer, events | MIT |
| Tailwind CSS (preflight styles) | MIT |

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
