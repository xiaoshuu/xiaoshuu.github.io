// src/utils/file-tree.ts
import { type CollectionEntry } from 'astro:content';

export interface FileNode {
  name: string;      // 显示在侧边栏的名字 (优先用 title，没有就用文件名)
  slug?: string;     // 点击跳转的路径
  date?: Date;       // 日期
  children: FileNode[];
  order?: number;    // 可选：排序权重
}

export function buildFileTree(notes: CollectionEntry<'notes'>[]): FileNode[] {
  const root: FileNode[] = [];

  notes.forEach((note) => {
    // 1. 解析文件名中的日期 (例如 2022-12-11-sort.md)
    // 如果 frontmatter 里没有 date，就尝试从文件名提取
    let noteDate = note.data.date;
    let fileName = note.slug.split('/').pop() || '';
    
    if (!noteDate) {
      const dateMatch = fileName.match(/^(\d{4}-\d{2}-\d{2})/);
      if (dateMatch) {
        noteDate = new Date(dateMatch[1]);
      }
    }

    // 2. 确定显示的标题 (优先用 frontmatter.title)
    const displayTitle = note.data.title || fileName.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/-/g, ' ');

    // 3. 构建路径
    const parts = note.slug.split('/');
    let currentLevel = root;

    parts.forEach((part, index) => {
      const isFile = index === parts.length - 1;
      
      // 检查当前层级是否已存在该节点
      //如果是文件夹，我们暂时用文件夹名；如果是文件，我们用 displayTitle
      const nodeName = isFile ? displayTitle : part.toUpperCase(); // 文件夹名字大写

      let existingNode = currentLevel.find(n => 
        // 这里的逻辑稍微复杂点：如果是文件夹，按文件夹名找；如果是文件，按 slug 找防止重名
        (isFile ? n.slug === `/notes/${note.slug}` : n.name === nodeName)
      );

      if (!existingNode) {
        existingNode = {
          name: nodeName,
          children: [],
          slug: isFile ? `/notes/${note.slug}` : undefined,
          date: isFile ? noteDate : undefined
        };
        currentLevel.push(existingNode);
      }

      currentLevel = existingNode.children;
    });
  });

  return sortTree(root);
}

// 排序：文件夹在最后（或者最前），文件按日期倒序
function sortTree(nodes: FileNode[]): FileNode[] {
  return nodes.sort((a, b) => {
    const aIsFolder = a.children.length > 0;
    const bIsFolder = b.children.length > 0;

    // 1. 文件夹排在后面 (个人习惯，你也可以改)
    if (aIsFolder && !bIsFolder) return 1;
    if (!aIsFolder && bIsFolder) return -1;

    // 2. 如果都是文件，按日期倒序 (新的在上面)
    if (!aIsFolder && !bIsFolder) {
      if (a.date && b.date) return b.date.valueOf() - a.date.valueOf();
    }

    return a.name.localeCompare(b.name);
  }).map(node => ({ ...node, children: sortTree(node.children) }));
}