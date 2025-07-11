"use client"

import { useState } from "react"

const personalityStocks = {
    Analytical: {
        stocks: ["MSFT", "GOOGL", "INTC"],
        reason: `You are logical, detail-oriented, and data-driven. 
      You prefer companies with strong fundamentals, transparent reporting, and long-term growth potential.
      Tech giants like Microsoft, Google, and Intel offer stability and data-rich business models that appeal to your mindset.`,
    },
    Driver: {
        stocks: ["TSLA", "NFLX", "AMZN"],
        reason: `You are ambitious, goal-driven, and decisive. 
      You’re comfortable with risk and seek high-growth opportunities. 
      Companies like Tesla, Netflix, and Amazon represent innovation, bold leadership, and aggressive market strategies — aligning with your forward-driving style.`,
    },
    Amiable: {
        stocks: ["JNJ", "PG", "KO"],
        reason: `You are loyal, consistent, and harmony-seeking. 
      You prioritize safety, long-term relationships, and steady performance. 
      Defensive stocks like Johnson & Johnson, Procter & Gamble, and Coca-Cola provide dependable dividends and low volatility — perfect for your temperament.`,
    },
    Expressive: {
        stocks: ["META", "NVDA", "UBER"],
        reason: `You are creative, enthusiastic, and vision-oriented. 
      You enjoy innovation and being part of exciting trends. 
      Companies like Meta (Facebook), Nvidia, and Uber drive the future with visionary products, making them ideal for your dynamic personality.`,
    },
} as const

type PersonalityType = keyof typeof personalityStocks

export default function RecommendPage() {
    const [selected, setSelected] = useState<PersonalityType>("Analytical")
    const current = personalityStocks[selected]

    return (
        <div className="min-h-screen bg-gray-50 py-12 px-6">
            <div className="max-w-4xl mx-auto bg-white p-8 rounded-xl shadow">
                <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
                    🧠 Personalized Stock Recommendations
                </h1>

                <div className="mb-6 text-center">
                    <label className="block text-lg font-medium mb-2 text-gray-700">
                        Select Your Personality Type:
                    </label>
                    <select
                        value={selected}
                        onChange={(e) => setSelected(e.target.value as PersonalityType)}
                        className="w-full md:w-1/2 border border-gray-300 rounded px-4 py-2 text-lg"
                    >
                        {Object.keys(personalityStocks).map((type) => (
                            <option key={type}>{type}</option>
                        ))}
                    </select>
                </div>

                <div className="mt-8">
                    <h2 className="text-2xl font-semibold text-green-700 mb-2">
                        Recommended Stocks:
                    </h2>
                    <ul className="list-disc list-inside text-xl text-green-800 mb-4">
                        {current.stocks.map((s) => (
                            <li key={s}>{s}</li>
                        ))}
                    </ul>
                    <p className="text-gray-700 leading-relaxed whitespace-pre-line text-lg">
                        {current.reason}
                    </p>
                </div>
            </div>
        </div>
    )
}
