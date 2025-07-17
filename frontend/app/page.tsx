"use client"

import Link from "next/link"

export default function HomePage() {
    return (
        <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-100 to-gray-300 p-8">
            <h1 className="text-5xl font-extrabold mb-6 text-gray-800 text-center tracking-wide">
                Start Your Real Investment
            </h1>
            <p className="text-gray-700 mb-10 text-center max-w-2xl text-lg leading-relaxed">
                You can freely access the data and forecasts of the stocks you are interested in,
                and also receive personalized investment advice.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl mb-10">
                <Link
                    href="/predict"
                    className="block bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6 border border-gray-200 hover:scale-105 transform duration-300"
                >
                    <h2 className="text-2xl font-bold text-blue-700 mb-2">🔍 Search & 🔮 Predict</h2>
                    <p className="text-gray-600">
                        View historical candlestick data and forecast future stock prices with AI models — all in one place.
                    </p>
                </Link>

                <Link
                    href="/recommend"
                    className="block bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6 border border-gray-200 hover:scale-105 transform duration-300"
                >
                    <h2 className="text-2xl font-bold text-green-700 mb-2">🧠 Personality-Based Picks</h2>
                    <p className="text-gray-600">
                        Choose your personality type and get stock suggestions tailored to your investment mindset.
                    </p>
                </Link>
            </div>

            <Link
                href="https://main-portal-three.vercel.app/"
                target="_blank"
                className="inline-block bg-gray-800 text-white px-6 py-3 rounded-full text-lg font-semibold shadow hover:bg-gray-700 transition"
            >
                🔙 Back to Project Portal
            </Link>
        </main>
    )
}
