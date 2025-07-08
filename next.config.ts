import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 你的其他配置项
  eslint: {
    ignoreDuringBuilds: true, // 忽略 ESLint 报错以避免部署失败
  },
};

export default nextConfig;
