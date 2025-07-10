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
} from "chart.js"
import "chartjs-chart-financial"
import { Chart } from "react-chartjs-2"
import "chartjs-adapter-date-fns"

import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

ChartJS.register(
  CategoryScale,
  TimeScale,
  LinearScale,
  Tooltip,
  Legend
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
}

export default function StockPage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [days, setDays] = useState(30)
  const [model, setModel] = useState("ensemble")
  const [showPrediction, setShowPrediction] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [candles, setCandles] = useState<CandleDataPoint[]>([])
  const [predictions, setPredictions] = useState<PredictionPoint[]>([])
  const [metrics, setMetrics] = useState<MetricInfo | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError("")
    setMetrics(null)
    setCandles([])
    setPredictions([])

    try {
      const [chartRes, predRes] = await Promise.all([
        axios.get(`/api/stock?symbol=${symbol}`),
        showPrediction
          ? axios.post("https://al-in-finance.onrender.com/api/predict", {
            symbol,
            days,
            model,
          })
          : Promise.resolve({ data: {} }),
      ])

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

      if (showPrediction && predRes.data?.predictions) {
        const lastDate = formattedCandles[formattedCandles.length - 1].x
        const preds: number[] = predRes.data.predictions
        const formattedPreds = preds.map((val, i) => {
          const d = new Date(lastDate)
          d.setDate(d.getDate() + i + 1)
          return { x: d, y: Number(val.toFixed(2)) }
        })
        setPredictions(formattedPreds)
        setMetrics(predRes.data || null)
      }
    } catch {
      setError("Failed to load data. Please try again.")
    } finally {
      setLoading(false)
    }
  }, [symbol, days, model, showPrediction])

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">📊 Stock Predictor</h1>

      <div className="flex gap-4 flex-wrap mb-6 items-end">
        <div>
          <Label htmlFor="symbol">Symbol</Label>
          <Input
            id="symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          />
        </div>
        <div>
          <Label htmlFor="days">Days to Predict</Label>
          <Input
            id="days"
            type="number"
            min={1}
            max={90}
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="model">Model</Label>
          <ToggleGroup
            type="single"
            value={model}
            onValueChange={(val) => val && setModel(val)}
            className="mt-1"
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
          <Label htmlFor="show-predict">Show Prediction</Label>
        </div>
        <Button onClick={fetchData} disabled={loading}>
          {loading ? "Loading..." : "Predict"}
        </Button>
      </div>

      {error && <p className="text-red-500">{error}</p>}

      {candles.length > 0 && (
        <div className="bg-white p-4 rounded shadow mb-6">
          <h2 className="text-lg font-medium mb-2">📈 Historical Candlestick</h2>
          <Chart
            type="candlestick"
            data={{
              datasets: [
                {
                  label: "OHLC",
                  data: candles.map((d) => ({
                    x: d.x,
                    o: d.o,
                    h: d.h,
                    l: d.l,
                    c: d.c,
                  })),
                  borderColor: "gray",
                  borderWidth: 1,
                },
              ],
            }}
            options={{
              responsive: true,
              scales: {
                x: {
                  type: "time",
                  time: { unit: "day" },
                  title: { display: true, text: "Date" },
                },
                y: {
                  title: { display: true, text: "Price (USD)" },
                },
              },
            }}
          />
        </div>
      )}

      {showPrediction && predictions.length > 0 && (
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-lg font-medium mb-2">🔮 Prediction</h2>
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
                x: {
                  type: "time",
                  time: { unit: "day" },
                  title: { display: true, text: "Date" },
                },
                y: {
                  title: { display: true, text: "Price (USD)" },
                },
              },
            }}
          />

          {metrics?.metrics && (
            <div className="mt-4 text-sm text-gray-700">
              <p>📊 Model: {metrics.metrics.model}</p>
              <p>📈 R² Score: {metrics.metrics.r2}</p>
              <p>📉 MAE: {metrics.metrics.mae}</p>
            </div>
          )}
          {metrics?.advice && (
            <div className="mt-2 text-sm text-blue-700">
              💡 <strong>Advice:</strong> {metrics.advice.suggestion}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
