// app/api/stock/route.ts

import { NextRequest, NextResponse } from "next/server"

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const symbol = searchParams.get("symbol") || "AAPL"

  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=3mo`

  try {
    const res = await fetch(url)

    if (!res.ok) {
      return NextResponse.json({ error: "Yahoo API responded with an error" }, { status: res.status })
    }

    const data: unknown = await res.json()
    return NextResponse.json(data)
  } catch {
    console.error("Yahoo fetch failed")
    return NextResponse.json({ error: "Failed to fetch Yahoo data" }, { status: 500 })
  }
}
