import { NextResponse } from "next/server";
import yahooFinance from "yahoo-finance2";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get("symbol");

  if (!symbol) {
    return NextResponse.json({ error: "No symbol provided" }, { status: 400 });
  }

  try {
    const result = await yahooFinance.historical(symbol, {
      period1: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      period2: new Date(),
    });

    return NextResponse.json(result);
  } catch {
    return NextResponse.json({ error: "Invalid symbol or API error" }, { status: 500 });
  }
}
