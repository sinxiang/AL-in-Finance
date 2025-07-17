"use client"

import { useState } from "react"

const personalityStocks = {
    Conservative: {
        stocks: ["JNJ", "PG", "KO"],
        reason: `You value capital preservation, stability, and low-risk investments.
Blue-chip companies like Johnson & Johnson, Procter & Gamble, and Coca-Cola offer strong balance sheets, consistent dividends, and defensive market positions — making them ideal for your conservative approach.`,
    },
    Balanced: {
        stocks: ["MSFT", "AAPL", "V"],
        reason: `You seek a healthy balance between risk and reward, favoring established companies with growth potential.
Microsoft, Apple, and Visa combine innovation, strong financials, and global presence — a balanced mix of growth and stability tailored for your investment style.`,
    },
    Aggressive: {
        stocks: ["TSLA", "NVDA", "AMZN"],
        reason: `You are comfortable with risk and driven by high-growth opportunities.
Tesla, Nvidia, and Amazon lead disruptive industries with rapid innovation and market dominance — aligning with your aggressive pursuit of returns.`,
    },
    Radical: {
        stocks: ["PLTR", "COIN", "ARKK"],
        reason: `You thrive on bold bets, emerging trends, and disruptive innovations.
Companies and funds like Palantir, Coinbase, and ARK Innovation ETF represent frontier technologies, high volatility, and speculative upside — matching your radical investment mindset.`,
    },
} as const

type PersonalityType = keyof typeof personalityStocks

export default function RecommendPage() {
    const [selected, setSelected] = useState<PersonalityType>("Conservative")
    const current = personalityStocks[selected]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-100 via-gray-200 to-gray-300 py-16 px-4 md:px-8 flex items-center justify-center">
            <div className="w-full max-w-3xl bg-white p-10 rounded-3xl shadow-2xl hover:shadow-md transition-all duration-300">
                <h1 className="text-4xl font-extrabold mb-8 text-center text-gray-800 tracking-wide">
                    🎯 Investment Personality Match
                </h1>

                <div className="mb-8 text-center">
                    <label className="block text-xl font-semibold mb-3 text-gray-700">
                        Select Your Investment Personality
                    </label>
                    <select
                        value={selected}
                        onChange={(e) => setSelected(e.target.value as PersonalityType)}
                        className="w-full md:w-2/3 border-2 border-gray-300 rounded-xl px-4 py-3 text-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-green-500 transition"
                    >
                        {Object.keys(personalityStocks).map((type) => (
                            <option key={type} value={type}>
                                {type}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="bg-gray-50 p-6 md:p-8 rounded-xl border border-gray-200 shadow-inner transition-all duration-300 hover:scale-[1.02]">
                    <h2 className="text-2xl font-bold text-green-700 mb-4 text-center">
                        📈 Top Stock Picks for {selected}
                    </h2>
                    <ul className="list-disc list-inside text-xl text-green-800 mb-6 space-y-2 text-center">
                        {current.stocks.map((s) => (
                            <li key={s} className="hover:underline hover:text-green-600 transition">
                                {s}
                            </li>
                        ))}
                    </ul>
                    <p className="text-gray-700 text-lg leading-relaxed whitespace-pre-line text-center">
                        {current.reason}
                    </p>
                </div>
            </div>
        </div>
    )
}
// 与之前相同，只在最下面加上这个：
<div className="text-center mt-8">
    <a
        href="https://al-in-finance.vercel.app/"
        target="_blank"
        className="inline-block bg-gray-800 text-white px-6 py-3 rounded-full text-lg font-semibold shadow hover:bg-gray-700 transition"
    >
        🔙 Back to Home
    </a>
</div>
