"use client"

import { useState } from "react"

// 推荐内容定义（类型安全）
const personalityStocks = {
    Analytical: {
        stocks: ["MSFT", "GOOGL", "INTC"],
        reason: "You prefer logic and data. These companies are tech-driven and stable.",
    },
    Driver: {
        stocks: ["TSLA", "NFLX", "AMZN"],
        reason: "You take risks and seek growth. These high-volatility stocks match your ambition.",
    },
    Amiable: {
        stocks: ["JNJ", "PG", "KO"],
        reason: "You value stability and harmony. These are long-term defensive plays.",
    },
    Expressive: {
        stocks: ["META", "NVDA", "UBER"],
        reason: "You enjoy innovation and social impact. These creative companies suit you well.",
    },
} as const

type PersonalityType = keyof typeof personalityStocks

export default function RecommendPage() {
    const [selected, setSelected] = useState<PersonalityType>("Analytical")
    const current = personalityStocks[selected]

    return (
        <div className="p-6 max-w-xl mx-auto">
            <h1 className="text-2xl font-semibold mb-4">🧠 Personality-Based Stock Picks</h1>

            <div className="mb-4">
                <label className="block font-medium mb-1">Select Your Personality Type:</label>
                <select
                    value={selected}
                    onChange={(e) => setSelected(e.target.value as PersonalityType)}
                    className="w-full border rounded p-2"
                >
                    {Object.keys(personalityStocks).map((type) => (
                        <option key={type}>{type}</option>
                    ))}
                </select>
            </div>

            <div className="bg-white p-4 rounded shadow">
                <h2 className="text-lg font-medium">Recommended Stocks:</h2>
                <ul className="list-disc list-inside text-green-700 my-2">
                    {current.stocks.map((s) => (
                        <li key={s}>{s}</li>
                    ))}
                </ul>
                <p className="text-gray-600">{current.reason}</p>
            </div>
        </div>
    )
}
