// app/api/predict/route.ts

import { NextRequest } from "next/server"

export async function GET(req: NextRequest): Promise<Response> {
  const symbol = req.nextUrl.searchParams.get("symbol") || "AAPL"

  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const url = `${backendUrl}/api/predict?symbol=${symbol}`

  try {
    const res = await fetch(url)

    if (!res.ok) {
      console.error(`Prediction API error: ${res.status} ${res.statusText}`)
      return new Response("Failed to fetch prediction", { status: res.status })
    }

    const data: unknown = await res.json()

    return new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    })
  } catch {
    console.error("Error contacting backend")
    return new Response("Error contacting backend", { status: 500 })
  }
}
