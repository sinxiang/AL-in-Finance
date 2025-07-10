"use client"

import React, { useState, useCallback, useEffect } from "react"
import axios from "axios"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"
import { Line } from "react-chartjs-2"
import "chartjs-adapter-date-fns"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Label } from "@/components/ui/label"

ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
)

type CandleDataPoint = {
  x: Date
  y: [number, number, number, number]
}

type PredictionDataPoint = {
  x: Date
  y: number
}

export default function StockChartPage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [days, setDays] = useState(30)
  const [model, setModel] = useState("ensemble")
  const [chartData, setChartData] = useState<CandleDataPoint[]>([])
  const [predictionData, setPredictionData] = useState<PredictionDataPoint[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError("")
      setMetrics(null)

      const [chartRes, predRes] = await Promise.all([
        axios.get(`/api/stock?symbol=${symbol}`),
        axios.post(
          `https://al-in-finance.onrender.com/api/predict`,
          { symbol, days, model },
          {
            headers: { "Content-Type": "application/json" },
            timeout: 20000,
          }
        ),
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
      setMetrics(predRes.data || null)
    } catch (err: any) {
      console.error("Error fetching data:", err)
      if (err.response?.data?.detail) {
        setError(`Server Error: ${err.response.data.detail}`)
      } else if (err.code === "ECONNABORTED") {
        setError("Request timed out. Server may be asleep.")
      } else {
        setError("Network error. Failed to fetch data.")
      }
    } finally {
      setLoading(false)
    }
  }, [symbol, days, model])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">📈 Stock Predictor</h1>

      <div className="flex gap-4 mb-6 items-end flex-wrap">
        <div>
          <Label htmlFor="symbol">Symbol</Label>
          <Input
            id="symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g. AAPL"
          />
        </div>
        <div>
          <Label htmlFor="days">Forecast Days</Label>
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
        <Button onClick={fetchData} disabled={loading}>
          {loading ? "Loading..." : "Predict"}
        </Button>
      </div>

      {error && <p className="text-red-600 mb-4">{error}</p>}

      <div className="bg-white shadow rounded-lg p-4">
        <Line
          data={{
            labels: [...chartData.map((d) => d.x), ...predictionData.map((d) => d.x)],
            datasets: [
              {
                label: "Close Price (Historical)",
                data: chartData.map((d) => d.y[3]),
                borderColor: "blue",
                backgroundColor: "rgba(0,0,255,0.1)",
                fill: false,
              },
              {
                label: "Predicted Close",
                data: [
                  ...new Array(chartData.length).fill(null),
                  ...predictionData.map((d) => d.y),
                ],
                borderColor: "green",
                backgroundColor: "rgba(0,255,0,0.1)",
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
      </div>

      {metrics?.metrics && (
        <div className="mt-6 text-sm text-gray-700 space-y-1">
          <p>📊 Model: {metrics.metrics.model}</p>
          <p>📈 R² Score: {metrics.metrics.r2}</p>
          <p>📉 MAE: {metrics.metrics.mae}</p>
        </div>
      )}

      {metrics?.advice && (
        <div className="mt-4 text-sm text-blue-700">
          💡 <b>Advice:</b> {metrics.advice.suggestion}
        </div>
      )}
    </div>
  )
}
