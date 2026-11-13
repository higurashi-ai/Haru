/** @type {import('tailwindcss').Config} */

// Haru's palette: white pages, soft grey cards, one red-orange accent.
export default {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      colors: {
        ink: "#1d1d1f",      // text
        muted: "#6e6e73",    // secondary text
        line: "#d2d2d7",     // field borders, silence dots
        paper: "#ffffff",    // page
        card: "#f5f5f7",     // grouped sections
        fill: "#e8e8ed",     // secondary buttons
        accent: "#e04a2f",   // the model lane's coral
        warning: "#b64400",
        lane: "#f4f5f7",
      },
    },
  },
  plugins: [],
};
