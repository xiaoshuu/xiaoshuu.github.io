// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://xiaoshuu.github.io',
  markdown: {
    // 允许 HTML 
    extendDefaultPlugins: true,     
    // 添加 remark 插件
    remarkPlugins: [remarkMath],   
    // 添加 rehype 插件
    rehypePlugins: [rehypeKatex],
  },
});
