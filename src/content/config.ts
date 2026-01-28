// src/content/config.ts
import { z, defineCollection } from 'astro:content';

const notesCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    // 允许 string 或 array，并统一转换为 array
    tags: z.union([z.string(), z.array(z.string())]).transform((val) => {
      return Array.isArray(val) ? val : [val];
    }).optional(),
    
    // 你的其他字段
    category: z.string().optional(), 
    // 兼容旧博客的 sidebar 字段，避免报错
    sidebar: z.object({
      nav: z.string().optional()
    }).optional(),
    // 兼容旧博客的 mathjax/aside 字段，避免报错（设为可选）
    mathjax: z.boolean().optional(),
    aside: z.object({
      toc: z.boolean().optional()
    }).optional(),
    key: z.string().optional(),
  }),
});

// 2. 定义“碎碎念”的格式
const thoughtsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date().optional(), // 碎碎念需要时间
  }),
});

// 3. 导出集合
export const collections = {
  'notes': notesCollection,
  'thoughts': thoughtsCollection,
  // 'tips': tipsCollection (同理)
};