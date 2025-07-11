// app/page.tsx
"use client"

import Link from "next/link"

export default function HomePage() {
    return (
        <main className="p-10 text-center">
            <h1 className="text-3xl font-bold mb-4">Welcome to Stock Web</h1>
            <p className="text-gray-600 mb-8">Choose a feature below to get started:</p>

            <div className="flex justify-center gap-6">
                <Link href="/search&predict" className="px-6 py-3 bg-blue-600 text-white rounded shadow">
                    📈 Stock Prediction
                </Link>
                <Link href="/recommend" className="px-6 py-3 bg-green-600 text-white rounded shadow">
                    🧠 Personality-Based Stock Picks
                </Link>
            </div>
        </main>
    )
}
