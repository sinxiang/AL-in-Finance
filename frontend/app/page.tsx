"use client"

import Link from "next/link"

export default function HomePage() {
    return (
        <main className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
            <h1 className="text-4xl font-bold mb-6 text-gray-800">Welcome to Stock Web</h1>
            <p className="text-gray-600 mb-10 text-center max-w-xl">
                Explore smart tools to analyze and select stocks using data, prediction, and personality alignment.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
                <Link href="/predict" className="block bg-white rounded-xl shadow-md hover:shadow-lg transition p-6 border border-gray-200">
                    <h2 className="text-2xl font-semibold text-blue-700 mb-2">🔍 Search & 🔮 Predict</h2>
                    <p className="text-gray-600">
                        View historical candlestick data and forecast future stock prices with AI models — all in one place.
                    </p>
                </Link>

                <Link href="/recommend" className="block bg-white rounded-xl shadow-md hover:shadow-lg transition p-6 border border-gray-200">
                    <h2 className="text-2xl font-semibold text-green-700 mb-2">🧠 Personality-Based Picks</h2>
                    <p className="text-gray-600">
                        Choose your personality type and get stock suggestions tailored to your investment mindset.
                    </p>
                </Link>
            </div>
        </main>
    )
}
