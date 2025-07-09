"use client"

import { useState, useEffect, useCallback } from "react"
import axios from "axios"
import dynamic from "next/dynamic"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false })

interface CandleDataPoint {
  x: Date
  y: [number, number, number, number]
}

interface PredictionDataPoint {
  x: Date
  y: number
}

export default function HomePage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [chartData, setChartData] = useState<CandleDataPoint[]>([])
  const [predictionData, setPredictionData] = useState<PredictionDataPoint[]>([])
  const [showPrediction, setShowPrediction] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError("")

      const [chartRes, predRes] = await Promise.all([
        axios.get(`/api/stock?symbol=${symbol}`),
        axios.get(`/api/predict?symbol=${symbol}`),
      ])

      const timestamps: number[] = chartRes.data.chart.result[0].timestamp
      const ohlc = chartRes.data.chart.result[0].indicators.quote[0]

      const formattedChartData: CandleDataPoint[] = timestamps.map((t, i) => ({
        x: new Date(t * 1000),
        y: [
          ohlc.open[i],
          ohlc.high[i],
          ohlc.low[i],
          ohlc.close[i],
        ].map((v) => Number(v.toFixed(2))) as [number, number, number, number],
      }))

      const futurePreds: number[] = predRes.data.predictions || []
      const lastDate = new Date(timestamps[timestamps.length - 1] * 1000)

      const formattedPredictionData: PredictionDataPoint[] = futurePreds.map((val, idx) => {
        const futureDate = new Date(lastDate)
        futureDate.setDate(futureDate.getDate() + idx + 1)
        return { x: futureDate, y: Number(val.toFixed(2)) }
      })

      setChartData(formattedChartData)
      setPredictionData(formattedPredictionData)
    } catch (err) {
      console.error("Error fetching data:", err)
      setError("Failed to fetch stock or prediction data.")
    } finally {
      setLoading(false)
    }
  }, [symbol])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return (
    <main className="p-6 max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">📈 Stock Chart with Prediction</h1>

      <div className="flex space-x-2">
        <Input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          placeholder="Enter Stock Symbol (e.g., AAPL)"
          className="max-w-xs"
        />
        <Button onClick={fetchData} disabled={loading}>
          {loading ? "Loading..." : "Load"}
        </Button>
      </div>

      <div className="flex items-center space-x-2">
        <input
          type="checkbox"
          checked={showPrediction}
          onChange={() => setShowPrediction(!showPrediction)}
        />
        <label>Show Prediction</label>
      </div>

      {error && <p className="text-red-500">{error}</p>}

      <section className="bg-white shadow rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-2">🕯️ Historical Candlestick Chart</h2>
        {chartData.length > 0 ? (
          <Chart
            type="candlestick"
            series={[{ data: chartData }]}
            options={{
              chart: { id: "candles", height: 350, type: "candlestick" },
              xaxis: { type: "datetime" },
              yaxis: { tooltip: { enabled: true } },
            }}
          />
        ) : (
          <p>No chart data</p>
        )}
      </section>

      {showPrediction && (
        <section className="bg-white shadow rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-2">🔮 Predicted Close Prices (Line Chart)</h2>
          {predictionData.length > 0 ? (
            <Chart
              type="line"
              series={[{ name: "Prediction", data: predictionData }]}
              options={{
                chart: { id: "prediction", height: 350 },
                xaxis: { type: "datetime" },
                yaxis: { decimalsInFloat: 2 },
              }}
            />
          ) : (
            <p>No prediction data</p>
          )}
        </section>
      )}
    </main>
  )
}
