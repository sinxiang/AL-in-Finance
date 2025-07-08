'use client';

import React, { useState } from 'react';

type StockData = {
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  adjClose?: number;
};

export default function Home() {
  const [symbol, setSymbol] = useState<string>("");
  const [prices, setPrices] = useState<StockData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const res = await fetch(`/api/stock?symbol=${symbol}`);
      const result = await res.json();

      if (result.error) {
        setError(result.error);
        setPrices([]);
        return;
      }

      setPrices(result.reverse());
      setError(null);
    } catch {
      setError("Error fetching stock data.");
      setPrices([]);
    }
  };

  return (
    <div style={{ padding: 40, fontFamily: "Arial" }}>
      <h1>📈 Stock Price Lookup</h1>
      <input
        type="text"
        placeholder="Enter stock symbol (e.g. AAPL)"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        style={{ padding: 8, width: 250, marginRight: 10 }}
      />
      <button onClick={fetchData} style={{ padding: "8px 16px" }}>
        Search
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {prices.length > 0 && (
        <table style={{ marginTop: 20, borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ border: "1px solid black", padding: 5 }}>Date</th>
              <th style={{ border: "1px solid black", padding: 5 }}>Close</th>
            </tr>
          </thead>
          <tbody>
            {prices.map((item) => (
              <tr key={item.date}>
                <td style={{ border: "1px solid black", padding: 5 }}>
                  {new Date(item.date).toLocaleDateString()}
                </td>
                <td style={{ border: "1px solid black", padding: 5 }}>
                  {item.close.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
