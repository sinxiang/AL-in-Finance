// app/api/predict/route.ts
import { NextRequest } from "next/server";

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol") || "AAPL";

  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const res = await fetch(`${backendUrl}/api/predict?symbol=${symbol}`);

    if (!res.ok) {
      return new Response("Failed to fetch prediction", { status: 500 });
    }

    const data = await res.json();
    return new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response("Error contacting backend", { status: 500 });
  }
}
