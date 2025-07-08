// app/api/stock/route.ts
import { NextRequest, NextResponse } from 'next/server';

const RAPIDAPI_KEY = '7f86051324mshf01077615510e7dp1ac9e4jsn2c4e365931d7';

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json({ error: 'Missing symbol' }, { status: 400 });
  }

  try {
    const url = `https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-timeseries?symbol=${symbol}&region=US`;

    const response = await fetch(url, {
      headers: {
        'X-RapidAPI-Key': RAPIDAPI_KEY,
        'X-RapidAPI-Host': 'apidojo-yahoo-finance-v1.p.rapidapi.com',
      },
      cache: 'no-store',
    });

    const raw = await response.json();

    const prices = raw?.prices;

    if (!Array.isArray(prices)) {
      return NextResponse.json({ error: 'Invalid data from API' }, { status: 500 });
    }

    // 格式化成图表用的数据结构
    const candles = prices
      .filter((item: any) => item.open && item.close && item.high && item.low && item.date)
      .map((item: any) => {
        const date = new Date(item.date * 1000); // UNIX 秒转毫秒
        return {
          time: date.toISOString().split('T')[0],
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        };
      })
      .slice(-30); // 只取最近 30 天

    return NextResponse.json(candles);
  } catch (error) {
    console.error('API fetch error:', error);
    return NextResponse.json({ error: 'Failed to fetch stock data' }, { status: 500 });
  }
}
