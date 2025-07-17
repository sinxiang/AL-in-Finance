"use client"

import Link from "next/link"

export default function HomePage() {
    return (
        <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-teal-100 via-cyan-100 to-blue-200 p-10">
            <h1 className="text-5xl font-extrabold mb-6 text-teal-900 text-center tracking-wide drop-shadow-sm">
                Start Your Real Investment
            </h1>
            <p className="text-teal-800 mb-12 text-center max-w-3xl text-lg leading-relaxed">
                You can freely access the data and forecasts of the stocks you are interested in,
                and also receive personalized investment advice.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-10 w-full max-w-5xl mb-12">
                <Link
                    href="/predict"
                    className="block bg-white rounded-3xl shadow-lg border border-teal-200 hover:shadow-2xl hover:border-cyan-400 transition p-8 hover:scale-105 transform duration-300"
                >
                    <h2 className="text-3xl font-bold text-cyan-700 mb-3">🔍 Search & 🔮 Predict</h2>
                    <p className="text-teal-700 text-lg leading-relaxed">
                        View historical candlestick data and forecast future stock prices with AI models — all in one place.
                    </p>
                </Link>

                <Link
                    href="/recommend"
                    className="block bg-white rounded-3xl shadow-lg border border-teal-200 hover:shadow-2xl hover:border-green-400 transition p-8 hover:scale-105 transform duration-300"
                >
                    <h2 className="text-3xl font-bold text-green-700 mb-3">🧠 Personality-Based Picks</h2>
                    <p className="text-teal-700 text-lg leading-relaxed">
                        Choose your personality type and get stock suggestions tailored to your investment mindset.
                    </p>
                </Link>
            </div>

            <Link
                href="https://main-portal-three.vercel.app/"
                target="_blank"
                className="inline-block bg-cyan-700 text-white px-8 py-4 rounded-full text-xl font-semibold shadow-lg hover:bg-cyan-800 transition"
            >
                🔙 Back to Project Portal
            </Link>
        </main>
    )
}
