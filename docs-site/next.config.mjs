import { createMDX } from 'fumadocs-mdx/next';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  outputFileTracingRoot: repoRoot,
  turbopack: {
    root: repoRoot,
  },
};

const withMDX = createMDX({
  configPath: 'source.config.ts',
});

export default withMDX(config);
