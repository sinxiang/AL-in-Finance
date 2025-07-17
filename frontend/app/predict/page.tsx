"use client"

import React, { useState, useCallback } from "react"
import axios from "axios"
import {
  Chart as ChartJS,
  TimeScale,
  LinearScale,
  Tooltip,
  Legend,
  CategoryScale,
  LineElement,
  PointElement,
  LineController,
  Filler,
} from "chart.js"
import {
  CandlestickController,
  CandlestickElement,
} from "chartjs-chart-financial"
import { Chart } from "react-chartjs-2"
import "chartjs-adapter-date-fns"

import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

import Link from "next/link"

ChartJS.register(
  TimeScale,
  LinearScale,
  Tooltip,
  Legend,
  CategoryScale,
  LineElement,
  PointElement,
  LineController,
  Filler,
  CandlestickController,
  CandlestickElement
)

type CandleDataPoint = {
  x: Date
  o: number
  h: number
  l: number
  c: number
}

type PredictionPoint = {
  x: Date
  y: number
}

type MetricInfo = {
  metrics?: {
    model: string
    r2: number
    mae: number
  }
  advice?: {
    trend: string
    risk: string
    suggestion: string
  }
  warning?: string
}

export default function PredictPage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [days, setDays] = useState(30)
  const [model, setModel] = useState("ensemble")
  const [showPrediction, setShowPrediction] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [warning, setWarning] = useState("")
  const [candles, setCandles] = useState<CandleDataPoint[]>([])
  const [predictions, setPredictions] = useState<PredictionPoint[]>([])
  const [metrics, setMetrics] = useState<MetricInfo | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError("")
    setWarning("")
    setMetrics(null)
    setCandles([])
    setPredictions([])

    try {
      const [chartRes, predRes] = await Promise.all([
        axios.get(`/api/stock?symbol=${symbol}`),
        showPrediction
          ? axios.post("https://al-in-finance-1.onrender.com/api/predict", {
            symbol,
            days,
            model,
          })
          : Promise.resolve({ data: {} }),
      ])

      if (!chartRes.data?.chart?.result || chartRes.data.chart.result.length === 0) {
        setError("No data found for this stock symbol.")
        return
      }

      const timestamps: number[] = chartRes.data.chart.result[0].timestamp
      const ohlc = chartRes.data.chart.result[0].indicators.quote[0]

      const formattedCandles: CandleDataPoint[] = timestamps.map((t, i) => ({
        x: new Date(t * 1000),
        o: Number(ohlc.open[i].toFixed(2)),
        h: Number(ohlc.high[i].toFixed(2)),
        l: Number(ohlc.low[i].toFixed(2)),
        c: Number(ohlc.close[i].toFixed(2)),
      }))
      setCandles(formattedCandles)

      if (showPrediction) {
        if (predRes.data?.predictions) {
          const lastDate = formattedCandles[formattedCandles.length - 1].x
          const preds: number[] = predRes.data.predictions
          const formattedPreds = preds.map((val, i) => {
            const d = new Date(lastDate)
            d.setDate(d.getDate() + i + 1)
            return { x: d, y: Number(val.toFixed(2)) }
          })
          setPredictions(formattedPreds)
          setMetrics(predRes.data || null)
        } else if (predRes.data?.warning) {
          setWarning(predRes.data.warning)
          setMetrics(predRes.data || null)
        }
      }
    } catch {
      setError("Failed to load data. Please try again.")
    } finally {
      setLoading(false)
    }
  }, [symbol, days, model, showPrediction])

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-100 via-cyan-100 to-blue-200 py-16 px-4 md:px-8 flex flex-col items-center">
      <div className="w-full max-w-5xl bg-white p-10 rounded-3xl shadow-2xl hover:shadow-md transition-all duration-300">
        <h1 className="text-4xl font-extrabold mb-4 text-center text-teal-900 tracking-wide">
          🔍 Search & 🔮 Predict Stock Data
        </h1>
        <p className="text-teal-800 mb-8 text-center text-lg leading-relaxed">
          Enter a stock symbol to view historical data and forecast future prices with different models.
        </p>

        <div className="bg-teal-50 p-4 rounded-xl border border-teal-200 shadow-inner mb-8 text-center text-teal-800 text-base leading-relaxed">
          ⚠️ The initial loading may take some time — please be patient. <br />
          ℹ️ To view historical data, simply enter the stock symbol and load the data. <br />
          If you want predictions, select a model, set the days, and turn on the prediction switch.
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-10">
          <div>
            <Label htmlFor="symbol" className="block mb-2 text-lg font-semibold text-teal-700">
              Stock Symbol
            </Label>
            <Input
              id="symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="text-lg px-4 py-3 border-teal-300 focus:ring-green-500 rounded-xl"
            />
          </div>

          <div>
            <Label htmlFor="days" className="block mb-2 text-lg font-semibold text-teal-700">
              Days to Predict (1-90)
            </Label>
            <Input
              id="days"
              type="number"
              min={1}
              max={90}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="text-lg px-4 py-3 border-teal-300 focus:ring-green-500 rounded-xl"
            />
          </div>
        </div>

        <div className="bg-teal-50 p-6 rounded-2xl border border-teal-200 shadow-inner mb-10">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <Label className="block mb-2 text-lg font-semibold text-teal-700">
                Prediction Model
              </Label>
              <ToggleGroup
                type="single"
                value={model}
                onValueChange={(val) => val && setModel(val)}
                className="flex flex-wrap gap-2"
              >
                <ToggleGroupItem value="ensemble">Ensemble</ToggleGroupItem>
                <ToggleGroupItem value="random_forest">RF</ToggleGroupItem>
                <ToggleGroupItem value="gb">GB</ToggleGroupItem>
                <ToggleGroupItem value="xgb">XGB</ToggleGroupItem>
                <ToggleGroupItem value="linear">Linear</ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                checked={showPrediction}
                onCheckedChange={setShowPrediction}
                id="show-predict"
              />
              <Label htmlFor="show-predict" className="text-lg text-teal-800">
                Show Prediction
              </Label>
            </div>

            <Button
              onClick={fetchData}
              disabled={loading}
              className="bg-cyan-700 text-white hover:bg-cyan-800 text-lg px-6 py-3 rounded-xl shadow"
            >
              {loading ? "Loading..." : "Load Data"}
            </Button>
          </div>
        </div>

        {error && <p className="text-red-500 mb-6">{error}</p>}

        {warning && (
          <div className="bg-yellow-100 text-yellow-800 p-4 rounded mb-6 text-center text-lg">
            ⚠️ {warning}
          </div>
        )}

        {candles.length > 0 && (
          <div className="bg-white p-6 rounded-2xl shadow mb-8">
            <h2 className="text-2xl font-bold text-green-700 mb-4 text-center">
              📊 Historical Candlestick
            </h2>
            <Chart
              type="candlestick"
              data={{
                datasets: [
                  {
                    label: "OHLC",
                    data: candles,
                    borderColor: candles.map((d) =>
                      d.c > d.o ? "#00b386" : d.c < d.o ? "#ff4d4f" : "#999"
                    ),
                    backgroundColor: candles.map((d) =>
                      d.c > d.o ? "#00b386" : d.c < d.o ? "#ff4d4f" : "#999"
                    ),
                    borderWidth: 1,
                    barThickness: 5,
                  },
                ],
              }}
              options={{
                responsive: true,
                plugins: { legend: { display: true }, tooltip: { enabled: true } },
                scales: {
                  x: { type: "time", time: { unit: "day" }, title: { display: true, text: "Date" } },
                  y: { title: { display: true, text: "Price (USD)" } },
                },
              }}
            />
          </div>
        )}

        {showPrediction && predictions.length > 0 && (
          <div className="bg-white p-6 rounded-2xl shadow mb-8">
            <h2 className="text-2xl font-bold text-green-700 mb-4 text-center">
              🔮 Prediction
            </h2>
            <Chart
              type="line"
              data={{
                labels: predictions.map((d) => d.x),
                datasets: [
                  {
                    label: "Predicted Close",
                    data: predictions.map((d) => d.y),
                    borderColor: "green",
                    fill: false,
                  },
                ],
              }}
              options={{
                responsive: true,
                scales: {
                  x: { type: "time", time: { unit: "day" }, title: { display: true, text: "Date" } },
                  y: { title: { display: true, text: "Price (USD)" } },
                },
              }}
            />

            {metrics?.metrics && (
              <div className="mt-4 text-gray-700 text-center text-lg">
                <p>📊 Model: {metrics.metrics.model}</p>
                <p>📈 R² Score: {metrics.metrics.r2}</p>
                <p>📉 MAE: {metrics.metrics.mae}</p>
              </div>
            )}
            {metrics?.advice && (
              <div className="mt-2 text-blue-700 text-center text-lg">
                💡 <strong>Advice:</strong> {metrics.advice.suggestion}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="text-center mt-8">
        <Link
          href="https://al-in-finance.vercel.app/"
          target="_blank"
          className="inline-block bg-cyan-700 text-white px-6 py-3 rounded-full text-lg font-semibold shadow hover:bg-cyan-800 transition"
        >
          🔙 Back to Home
        </Link>
      </div>
    </div>
  )
}
