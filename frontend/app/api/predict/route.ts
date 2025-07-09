// app/api/predict/route.ts

import { NextRequest, NextResponse } from "next/server"

export async function GET(req: NextRequest): Promise<NextResponse> {
  const symbol = req.nextUrl.searchParams.get("symbol") || "AAPL"
  const days = req.nextUrl.searchParams.get("days") || "5"

  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const url = `${backendUrl}/api/predict`

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, days: Number(days) }),
    })

    if (!res.ok) {
      console.error(`Prediction API error: ${res.status} ${res.statusText}`)
      return new NextResponse("Failed to fetch prediction", { status: res.status })
    }

    const data = await res.json()

    return NextResponse.json(data)
  } catch (err) {
    console.error("Error contacting backend:", err)
    return new NextResponse("Error contacting backend", { status: 500 })
  }
}
