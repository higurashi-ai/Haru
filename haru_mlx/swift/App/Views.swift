// Screens in Haru's style: white pages, soft grey rounded cards, capsule
// buttons and one red-orange accent, the coral of the model's lane. The orbs and waveform lanes (Waveform.swift)
// draw at screen rate inside TimelineView, so they never re-render the rest
// of the window.

import SwiftUI

// MARK: - Theme

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255, opacity: opacity)
    }
}

enum Theme {
    static let ink = Color(hex: 0x1d1d1f)      // text
    static let muted = Color(hex: 0x6e6e73)    // secondary text
    static let line = Color(hex: 0xd2d2d7)     // field borders, silence dots
    static let paper = Color.white             // page
    static let card = Color(hex: 0xf5f5f7)     // grouped sections, lanes
    static let fill = Color(hex: 0xe8e8ed)     // secondary buttons
    static let accent = Color(hex: 0xe04a2f)   // the model lane's coral, dark enough for white text on it
    static let success = accent
    static let warning = Color(hex: 0xb64400)
}

/// Capsule buttons: red-orange and filled for the main action (`accent`) or a
/// chosen option (`selected`), grey otherwise.
struct HaruButtonStyle: ButtonStyle {
    var selected = false
    var accent = false
    var large = false

    func makeBody(configuration: Configuration) -> some View {
        CapsuleLabel(configuration: configuration, filled: selected || accent, large: large)
    }

    private struct CapsuleLabel: View {
        let configuration: ButtonStyleConfiguration
        let filled: Bool
        let large: Bool
        @Environment(\.isEnabled) private var enabled

        var body: some View {
            configuration.label
                .font(.system(size: large ? 15 : 13, weight: large ? .semibold : .medium))
                .padding(.vertical, large ? 11 : 6)
                .padding(.horizontal, large ? 22 : 14)
                .foregroundStyle(filled ? Color.white : Theme.ink)
                .background(Capsule().fill(filled ? Theme.accent : Theme.fill))
                .opacity(enabled ? (configuration.isPressed ? 0.75 : 1) : 0.4)
                .contentShape(Capsule())
        }
    }
}

/// A grouped section: soft grey, rounded, like Settings.
struct Card<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 16) { content }
            .padding(22)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(Theme.card))
    }
}

struct FieldBox<Content: View>: View {
    let label: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.muted)
            content
        }
    }
}

/// White, rounded, with a blue ring while focused.
private struct FieldChrome: ViewModifier {
    var focused = false

    func body(content: Content) -> some View {
        content
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.paper))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(focused ? Theme.accent : Theme.line, lineWidth: focused ? 2 : 1))
    }
}

struct HaruTextField: View {
    let label: String
    @Binding var text: String
    @FocusState private var focused: Bool

    var body: some View {
        FieldBox(label: label) {
            TextField("", text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .focused($focused)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .modifier(FieldChrome(focused: focused))
        }
    }
}

struct HaruTextEditor: View {
    let label: String
    @Binding var text: String
    var height: CGFloat = 80
    @FocusState private var focused: Bool

    var body: some View {
        FieldBox(label: label) {
            TextEditor(text: $text)
                .font(.system(size: 14))
                .scrollContentBackground(.hidden)
                .focused($focused)
                .padding(8)
                .frame(minHeight: height)
                .modifier(FieldChrome(focused: focused))
        }
    }
}

struct HaruSelect: View {
    let options: [(String, String)]
    @Binding var selection: String

    var body: some View {
        // The whole field opens the menu: a plain button-style menu draws this
        // label as is, where a borderless one only answers clicks on its text.
        Menu {
            ForEach(options, id: \.0) { tag, label in Button(label) { selection = tag } }
        } label: {
            HStack(spacing: 8) {
                Text(options.first { $0.0 == selection }?.1 ?? selection).font(.system(size: 14))
                    .foregroundStyle(Theme.ink).lineLimit(1)
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.muted)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, minHeight: 38)
            .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .modifier(FieldChrome())
    }
}

struct SectionTitle: View {
    let title: String
    let note: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.system(size: 20, weight: .semibold))
            if !note.isEmpty {
                Text(note).font(.system(size: 13)).foregroundStyle(Theme.muted)
            }
        }
    }
}

/// The model's orb at rest, as a logo.
struct IdleOrb: View {
    @State private var state = BlobState()

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { ctx, size in
                BlobPainter.draw(ctx, size: size, level: 0, time: timeline.date.timeIntervalSinceReferenceDate,
                                 state: state, colors: Palette.model)
            }
        }
    }
}

struct Masthead: View {
    var compact = false

    var body: some View {
        if compact {
            HStack(spacing: 6) {
                IdleOrb().frame(width: 30, height: 30)
                Text("Haru").font(.system(size: 22, weight: .semibold)).tracking(-0.3)
            }
        } else {
            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    // The orb fills about the middle half of its canvas: lay it
                    // out at that width so the pair is centred as it looks.
                    IdleOrb().frame(width: 84, height: 84).padding(.horizontal, -20)
                    Text("Haru").font(.system(size: 52, weight: .semibold)).tracking(-1)
                }
                Text("Full-Duplex Speech-to-Speech AI").font(.system(size: 20, weight: .medium))
                    .foregroundStyle(Theme.muted)
            }
        }
    }
}

// MARK: - Root

struct ContentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ZStack {
            Theme.paper.ignoresSafeArea()
            switch model.phase {
            case .launching: LaunchView()
            case .ready, .starting: SetupView()
            case .talking: ConversationView()
            }
        }
        .foregroundStyle(Theme.ink)
        .tint(Theme.accent)
        .alert("エラー", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

struct LaunchView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 24) {
            Masthead()
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(model.status).foregroundStyle(Theme.muted)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(model.serverLog.enumerated()), id: \.offset) { i, line in
                            Text(line).font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.muted).id(i)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(14)
                }
                .onChange(of: model.serverLog.count) { n in proxy.scrollTo(n - 1, anchor: .bottom) }
            }
            .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(Theme.card))
        }
        .padding(32)
        .frame(maxWidth: 760)
    }
}

// MARK: - Setup

struct SetupView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Masthead().frame(maxWidth: .infinity).padding(.top, 12).padding(.bottom, 8)
                startRow.padding(.bottom, 8)
                if (model.lastReport?.frames ?? 0) > 0 || model.lastRecording != nil { lastConversation }
                cardSection
                referenceSection
                settingsSection
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: 720)
            .frame(maxWidth: .infinity)
        }
    }

    private var lastConversation: some View {
        Card {
            if let r = model.lastReport, r.frames > 0 {
                HStack(alignment: .bottom, spacing: 28) {
                    Stat(label: "前回の応答遅延", value: String(format: "%.0f ms", r.p50 * 1000), big: true)
                    Stat(label: "p95", value: String(format: "%.0f ms", r.p95 * 1000))
                    Spacer()
                }
            }
            if model.lastRecording != nil { saveRow }
        }
    }

    /// The last conversation's audio and video, nothing else.
    private var saveRow: some View {
        HStack(spacing: 10) {
            Text("前回の会話を保存").font(.system(size: 13)).foregroundStyle(Theme.muted)
            Button { model.saveAudio() } label: { Label("音声", systemImage: "waveform") }
                .buttonStyle(HaruButtonStyle())
            Button { model.saveVideo() } label: { Label("動画", systemImage: "film") }
                .buttonStyle(HaruButtonStyle())
                .disabled(model.exportProgress != nil)
            if let p = model.exportProgress {
                ProgressView(value: p).frame(width: 120)
                Text("書き出しています… \(Int(p * 100))%")
                    .font(.system(size: 12)).monospacedDigit().foregroundStyle(Theme.muted)
            }
            Spacer()
        }
    }

    private var cardSection: some View {
        Card {
            HStack(alignment: .top) {
                SectionTitle(title: "キャラクター", note: "モデルの人物設定。空欄の項目は省略されます。")
                Spacer()
                HStack(spacing: 6) {
                    ForEach(model.presets) { preset in
                        Button(preset.label) { model.applyPreset(preset) }
                            .buttonStyle(HaruButtonStyle(selected: model.card == CardFields(preset.card)))
                    }
                }
            }
            FieldBox(label: "言語（モデルが話す言語。見出しと既定の声もこの言語になります）") {
                HaruSelect(options: model.languages.map { ($0.code, $0.label) },
                            selection: Binding(get: { model.language }, set: { model.selectLanguage($0) }))
            }
            // Labels from your side; the model's prompt keeps the
            // template's own headers (its name as "your name", yours as "the other's").
            HStack(alignment: .top, spacing: 14) {
                HaruTextField(label: "モデルの名前", text: $model.card.name)
                HaruTextField(label: "あなたの名前", text: $model.card.otherName)
            }
            HStack(alignment: .top, spacing: 14) {
                FieldBox(label: "モデルの性別") {
                    HaruSelect(options: [("", "指定しない"), ("female", "女性"), ("male", "男性")],
                                selection: $model.card.gender)
                }
                FieldBox(label: "モデルの相づち") {
                    HaruSelect(options: [("", "指定しない"), ("low", "少なめ"), ("mid", "普通"), ("high", "多め")],
                                selection: $model.card.backchannel)
                }
            }
            HaruTextEditor(label: "モデルの人物設定（必須）", text: $model.card.persona, height: 110)
            HaruTextEditor(label: "モデルとあなたの関係性", text: $model.card.relationship, height: 64)
            HaruTextEditor(label: "状況", text: $model.card.situation, height: 64)
        }
    }

    private var referenceSection: some View {
        Card {
            SectionTitle(title: "声", note: "モデルの声の参照音声。1 人の話者で 3 秒〜3 分。")
            HStack(spacing: 6) {
                ForEach(ReferenceSource.allCases) { source in
                    Button(source.rawValue) { model.referenceSource = source }
                        .buttonStyle(HaruButtonStyle(selected: model.referenceSource == source))
                }
                Spacer()
                Button { model.playReference() } label: { Label("試聴", systemImage: "play.fill") }
                    .buttonStyle(HaruButtonStyle())
                    .disabled(model.referenceURL == nil)
            }
            HStack(spacing: 10) {
                switch model.referenceSource {
                case .preset:
                    Text("既定の声（\(model.languages.first { $0.code == model.language }?.label ?? "")）")
                        .foregroundStyle(Theme.muted)
                case .file:
                    Text(model.referenceFile?.lastPathComponent ?? "ファイルが選ばれていません")
                        .foregroundStyle(Theme.muted)
                    Spacer()
                    Button("ファイルを選択…") { model.chooseFile() }.buttonStyle(HaruButtonStyle())
                case .recording:
                    if model.isRecording {
                        Circle().fill(Color(hex: 0xff3b30)).frame(width: 9, height: 9)
                    }
                    Text(model.recordingURL == nil ? "まだ録音していません"
                         : String(format: "%.1f 秒", model.recordingSeconds))
                        .monospacedDigit().foregroundStyle(Theme.muted)
                    Spacer()
                    Button(model.isRecording ? "録音を停止" : "録音を開始") { model.toggleRecording() }
                        .buttonStyle(HaruButtonStyle(selected: model.isRecording))
                }
            }
            .font(.system(size: 13))
            .frame(minHeight: 28)
        }
    }

    private var settingsSection: some View {
        Card {
            SectionTitle(title: "設定", note: "")
            HStack(alignment: .top, spacing: 14) {
                FieldBox(label: "マイク") {
                    HaruSelect(options: model.inputDevices.map { ($0, $0) }, selection: $model.inputDevice)
                }
                FieldBox(label: "出力") {
                    HaruSelect(options: model.outputDevices.map { ($0, $0) }, selection: $model.outputDevice)
                }
            }
            deviceAdvice
            Button("デバイスを再検出") { model.refreshDevices() }
                .buttonStyle(.plain).font(.system(size: 13)).foregroundStyle(Theme.accent)
            // Each control sits between what its two ends do.
            HStack(spacing: 10) {
                settingLabel("CFG scale")
                hint("自然")
                Slider(value: $model.cfgScale, in: 1.0...2.0, step: 0.05).frame(width: 200)
                hint("はっきり")
                Text(String(format: "%.2f", model.cfgScale)).monospacedDigit().padding(.leading, 6)
            }
            HStack(spacing: 10) {
                settingLabel("マージン")
                hint("高速")
                Button { model.marginMs = max(5, model.marginMs - 5) } label: { Image(systemName: "minus") }
                    .buttonStyle(HaruButtonStyle())
                Text("\(Int(model.marginMs)) ms").monospacedDigit().frame(width: 52)
                Button { model.marginMs = min(40, model.marginMs + 5) } label: { Image(systemName: "plus") }
                    .buttonStyle(HaruButtonStyle())
                hint("安定")
            }
            HStack(spacing: 10) {
                settingLabel("精度")
                hint("高速")
                ForEach([("int8", "int8"), ("bf16", "BF16")], id: \.0) { value, label in
                    Button(label) { model.setPrecision(value) }
                        .buttonStyle(HaruButtonStyle(selected: (model.precision.isEmpty ? "int8" : model.precision) == value))
                }
                hint("高精度")
            }
            .disabled(model.isRecording || model.exportProgress != nil)
        }
    }

    private func settingLabel(_ text: String) -> some View {
        Text(text).font(.system(size: 13, weight: .medium)).frame(width: 84, alignment: .leading)
    }

    private func hint(_ text: String) -> some View {
        Text(text).font(.system(size: 12)).foregroundStyle(Theme.muted)
    }

    private var deviceAdvice: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "headphones").font(.system(size: 15)).foregroundStyle(Theme.muted)
            VStack(alignment: .leading, spacing: 6) {
                Text("有線のヘッドホンと、有線または USB のマイクがおすすめです").font(.system(size: 13, weight: .semibold))
                Text("""
                    デバイスそのものの遅延が応答の遅延に加わります。MacBook の内蔵スピーカーは約 20 ms \
                    （有線ヘッドホンは約 8 ms）、内蔵マイクは約 30 ms、Bluetooth（AirPods など）は 100 ms 以上です。
                    """)
                    .font(.system(size: 12)).foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
                if let warning = model.deviceWarning {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.system(size: 12)).foregroundStyle(Theme.warning)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.paper))
    }

    private var startRow: some View {
        HStack(spacing: 14) {
            Spacer()
            prepareStatus
            Button(model.phase == .starting ? "接続中…" : "会話を始める") { model.startConversation() }
                .buttonStyle(HaruButtonStyle(accent: true, large: true))
                .keyboardShortcut(.defaultAction)
                .disabled(model.phase != .ready || model.isRecording || model.exportProgress != nil)
        }
    }

    /// Whether the server has prefilled the current card and voice.
    @ViewBuilder private var prepareStatus: some View {
        let small = Font.system(size: 12)
        if model.phase == .starting || model.prepare == .preparing {
            ProgressView().controlSize(.small)
            Text("準備中…").font(small).foregroundStyle(Theme.muted)
        } else if model.prepare == .ready {
            Label("準備完了", systemImage: "checkmark.circle.fill").font(small).foregroundStyle(Theme.success)
        } else if case .failed(let message) = model.prepare {
            Label(message, systemImage: "exclamationmark.triangle.fill").font(small)
                .foregroundStyle(Theme.warning).lineLimit(2)
        }
    }
}

struct Stat: View {
    let label: String
    let value: String
    var big = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.muted)
            Text(value).font(.system(size: big ? 26 : 16, weight: big ? .semibold : .regular))
                .foregroundStyle(big ? Theme.accent : Theme.ink)
                .monospacedDigit()
        }
    }
}

// MARK: - Conversation

struct ConversationView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 20) {
            HStack(spacing: 12) {
                Button("会話を終える") { model.stopConversation() }
                    .buttonStyle(HaruButtonStyle())
                    .keyboardShortcut(.cancelAction)
                Circle().fill(Theme.success).frame(width: 8, height: 8)
                Text("接続中").font(.system(size: 13)).foregroundStyle(Theme.muted)
                Spacer()
                Masthead(compact: true)
            }
            stats
            WaveLanes(levels: model.levels, waves: model.waves,
                      userName: model.card.otherName.isEmpty ? "あなた" : model.card.otherName,
                      modelName: model.card.name.isEmpty ? "モデル" : model.card.name)
            Subtitles(waves: model.waves)
                .padding(.leading, 126)
                .frame(maxHeight: .infinity)
        }
        .padding(24)
    }

    private var stats: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .bottom, spacing: 28) {
                Stat(label: "応答の遅延", value: model.latencyMs.map { String(format: "%.0f ms", $0) } ?? "—",
                     big: true)
                Stat(label: "LLM ステップ", value: "\(Int(model.stepMs)) ms")
                Stat(label: "入出力デバイス", value: model.audioInfo)
                Spacer()
            }
        }
        .padding(18)
        .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(Theme.card))
    }
}
