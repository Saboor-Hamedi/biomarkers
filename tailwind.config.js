/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/renderer/index.html",
    "./src/renderer/src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          dark: '#0e1117',
          card: '#ffffff',
          accent: '#1e293b'
        }
      }
    },
  },
  plugins: [],
}
