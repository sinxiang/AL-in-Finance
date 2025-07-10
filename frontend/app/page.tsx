"use client"

import { useState, useEffect, useCallback } from "react"
import axios from "axios"
import dynamic from "next/dynamic"
import { Input } from "../components/ui/input"
import { Button } from "../components/ui/button"

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false })

interface CandleDataPoint {
  x: Date
  y: [number, number, number, number]
}

interface PredictionDataPoint {
  x: Date
  y: number
}

interface Metrics {
  model: string
  r2: number
  mae: number
}

export default function HomePage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [days, setDays] = useState(30)
  const [model, setModel] = useState("ensemble")
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [chartData, setChartData] = useState<CandleDataPoint[]>([])
  const [predictionData, setPredictionData] = useState<PredictionDataPoint[]>([])
  const [showPrediction, setShowPrediction] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError("")
      setMetrics(null)

      const [chartRes, predRes] = await Promise.all([
        axios.get(`/api/stock?symbol=${symbol}`),
        axios.post(`https://al-in-finance.onrender.com/api/predict`, {
          symbol,
          days,
          model,
        }),
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
      setMetrics(predRes.data.metrics || null)
    } catch (err) {
      console.error("Error fetching data:", err)
      setError("Failed to fetch stock or prediction data.")
    } finally {
      setLoading(false)
    }
  }, [symbol, days, model])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return (
    <main className="p-6 max-w-5xl mx-auto space-y-8">
      <h1 className="text-3xl font-extrabold text-center text-gray-800">📈 Stock Forecast Dashboard</h1>

      <section className="bg-white shadow-lg rounded-2xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row flex-wrap items-center gap-4">
          <Input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter Stock Symbol (e.g., AAPL)"
            className="w-full sm:max-w-xs border-gray-300"
          />
          <Input
            type="number"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            placeholder="Days to Predict"
            className="w-full sm:max-w-xs border-gray-300"
          />
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full sm:max-w-xs border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="ensemble">Ensemble (Avg)</option>
            <option value="random_forest">Random Forest</option>
            <option value="gb">Gradient Boosting</option>
            <option value="linear">Linear Regression</option>
            <option value="xgb">XGBoost</option>
          </select>
          <Button onClick={fetchData} disabled={loading} className="w-full sm:w-auto">
            {loading ? "Loading..." : "Load"}
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={showPrediction}
            onChange={() => setShowPrediction(!showPrediction)}
            className="w-4 h-4"
          />
          <label className="text-sm text-gray-700">Show Prediction</label>
        </div>

        {error && <p className="text-red-500 font-medium">{error}</p>}
      </section>

      <section className="bg-white shadow-md rounded-xl p-6">
        <h2 className="text-lg font-semibold mb-4">🕯️ Historical Candlestick Chart</h2>
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
          <p className="text-sm text-gray-500">No chart data</p>
        )}
      </section>

      {showPrediction && (
        <section className="bg-white shadow-md rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">🔮 Predicted Close Prices</h2>
          {predictionData.length > 0 ? (
            <>
              <Chart
                type="line"
                series={[{ name: `Prediction (${model})`, data: predictionData }]}
                options={{
                  chart: { id: "prediction", height: 350 },
                  xaxis: { type: "datetime" },
                  yaxis: { decimalsInFloat: 2 },
                }}
              />
              {metrics && (
                <div className="text-sm text-gray-600 mt-4">
                  📊 Model: <b>{metrics.model}</b> | R²: <b>{metrics.r2}</b> | MAE: <b>{metrics.mae}</b>
                </div>
              )}
            </>
          ) : (
            <p className="text-sm text-gray-500">No prediction data</p>
          )}
        </section>
      )}
    </main>
  )
}
