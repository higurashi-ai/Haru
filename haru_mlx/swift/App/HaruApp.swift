// macOS app for Haru: native audio I/O, the model in
// a Python server process that the app starts when needed.

import SwiftUI

@main
struct HaruApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("Haru") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 920, minHeight: 760)
                .preferredColorScheme(.light)
                .onAppear { model.launch() }
        }
    }
}
