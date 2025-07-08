'use client';

import React, { useEffect, useRef, useState } from 'react';
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
      const response = await fetch(
        `https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-chart?interval=1d&symbol=${symbol}&range=1mo&region=US`,
        {
          method: 'GET',
          headers: {
            'X-RapidAPI-Key': '7f86051324mshf01077615510e7dp1ac9e4jsn2c4e365931d7',
            'X-RapidAPI-Host': 'apidojo-yahoo-finance-v1.p.rapidapi.com',
          },
        }
      );

      const data = await response.json();

      const result = data.chart?.result?.[0];
      if (!result) throw new Error('Invalid symbol');

      const timestamps = result.timestamp;
      const quotes = result.indicators.quote[0];

      const candleData: Candle[] = timestamps.map((t: number, i: number) => ({
        time: new Date(t * 1000).toISOString().split('T')[0] as Time,
        open: quotes.open[i],
        high: quotes.high[i],
        low: quotes.low[i],
        close: quotes.close[i],
      }));

      setCandles(candleData);
      setError(null);
    } catch (err) {
      console.error(err);
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
