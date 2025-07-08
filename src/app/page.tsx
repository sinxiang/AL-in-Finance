'use client';

import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip } from 'chart.js';
import { Chart } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

export default function HomePage() {
  const [symbol, setSymbol] = useState('');
  const [candles, setCandles] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setError(null);
    try {
      const res = await fetch(`/api/stock?symbol=${symbol}`);
      const raw = await res.json();

      if (!Array.isArray(raw)) {
        setError('Invalid data format from API');
        return;
      }

      const cleaned = raw.map((d: any) => ({
        date: d.date.split('T')[0],
        open: d.open,
        close: d.close,
        high: d.high,
        low: d.low,
      }));

      setCandles(cleaned);
    } catch (err) {
      setError('Failed to fetch data.');
    }
  };

  const chartData = {
    labels: candles.map((d) => d.date),
    datasets: [
      {
        label: 'Open',
        data: candles.map((d) => d.open),
        backgroundColor: 'blue',
      },
      {
        label: 'Close',
        data: candles.map((d) => d.close),
        backgroundColor: 'green',
      },
    ],
  };

  return (
    <main className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">📊 Simple Stock Chart</h1>

      <div className="flex gap-2 mb-4">
        <input
          className="border px-2 py-1"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          placeholder="Enter symbol (e.g. AAPL)"
        />
        <button className="bg-blue-600 text-white px-4 py-1 rounded" onClick={fetchData}>
          Fetch
        </button>
      </div>

      {error && <p className="text-red-500">{error}</p>}

      {candles.length > 0 && (
        <div className="bg-white p-4 rounded shadow">
          <Chart type="bar" data={chartData} />
        </div>
      )}
    </main>
  );
}
