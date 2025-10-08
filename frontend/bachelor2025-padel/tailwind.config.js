/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ["Orbitron", "ui-sans-serif", "system-ui"],
            },
            screens: {
                'xs': '640px',
                'desktop': '1024px',
            },
        },
    },
    plugins: [],
};