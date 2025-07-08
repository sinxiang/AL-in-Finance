'use client';

import React, { useEffect, useRef, useState } from 'react';
import yahooFinance from 'yahoo-finance2';
import { createChart, Time } from 'lightweight-charts';

type Candle = {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
};

export default function Home() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [symbol, setSymbol] = useState('');
  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const result = await yahooFinance.historical(symbol, {
        period1: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
        period2: new Date(),
      });

      const candleData: Candle[] = result.reverse().map((d) => ({
        time: d.date.toISOString().split('T')[0] as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));

      setCandles(candleData);
      setError(null);
    } catch {
      setError('Invalid stock symbol or API error.');
      setCandles([]);
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current || candles.length === 0) return;

    chartContainerRef.current.innerHTML = '';

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: { background: { color: '#ffffff' }, textColor: '#000' },
      grid: {
        vertLines: { color: '#eee' },
        horzLines: { color: '#eee' },
      },
    });

    const series = (chart as any).addCandlestickSeries();

    series.setData(candles);

    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: chartContainerRef.current!.clientWidth });
    });
    resizeObserver.observe(chartContainerRef.current);

    return () => resizeObserver.disconnect();
  }, [candles]);

  return (
    <div style={{ padding: 40, fontFamily: 'Arial', maxWidth: 900, margin: 'auto' }}>
      <h1 style={{ fontSize: '2rem', marginBottom: 20 }}>📊 Stock Candlestick Chart</h1>

      <div style={{ display: 'flex', marginBottom: 20 }}>
        <input
          type="text"
          placeholder="Enter symbol (e.g. AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          style={{
            padding: 10,
            width: 300,
            marginRight: 10,
            border: '1px solid #ccc',
            borderRadius: 5,
          }}
        />
        <button
          onClick={fetchData}
          style={{
            padding: '10px 20px',
            backgroundColor: '#0070f3',
            color: 'white',
            border: 'none',
            borderRadius: 5,
            cursor: 'pointer',
          }}
        >
          Search
        </button>
      </div>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <div ref={chartContainerRef} />
    </div>
  );
}
