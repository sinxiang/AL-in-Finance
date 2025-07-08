"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import axios from "axios";

// 动态加载 ApexCharts（避免 SSR 报错）
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

export default function Home() {
  const [symbol, setSymbol] = useState("AAPL");
  const [input, setInput] = useState("AAPL");
  const [series, setSeries] = useState<any[]>([]);
  const [options, setOptions] = useState<any>({
    chart: {
      type: "candlestick",
      height: 350,
    },
    title: {
      text: "Stock Candlestick Chart",
      align: "left",
    },
    xaxis: {
      type: "datetime",
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
    },
  });

  const fetchData = async (ticker: string) => {
    try {
      const res = await axios.get(`/api/stock?symbol=${ticker}`);
      const result = res.data.chart.result?.[0];

      if (!result) throw new Error("No data returned");

      const timestamps = result.timestamp;
      const ohlc = result.indicators.quote[0];

      const formattedData = timestamps.map((t: number, i: number) => ({
        x: new Date(t * 1000),
        y: [
          ohlc.open[i],
          ohlc.high[i],
          ohlc.low[i],
          ohlc.close[i],
        ].map((v) => parseFloat(v.toFixed(2))),
      }));

      setSeries([{ data: formattedData }]);
      setOptions((prev: any) => ({
        ...prev,
        title: { text: `${ticker.toUpperCase()} Stock Candlestick Chart` },
      }));
    } catch (err) {
      console.error("Failed to fetch data:", err);
      setSeries([]);
    }
  };

  useEffect(() => {
    fetchData(symbol);
  }, [symbol]);

  return (
    <main className="p-6">
      <h1 className="text-2xl font-bold mb-4">Stock Candle Chart Viewer</h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSymbol(input.trim().toUpperCase());
        }}
        className="flex gap-2 mb-6"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="border px-4 py-2 rounded w-40"
          placeholder="Enter stock symbol"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Search
        </button>
      </form>

      {series.length > 0 ? (
        <ApexChart
          options={options}
          series={series}
          type="candlestick"
          height={350}
        />
      ) : (
        <p className="text-gray-500">No data available for {symbol}.</p>
      )}
    </main>
  );
}
