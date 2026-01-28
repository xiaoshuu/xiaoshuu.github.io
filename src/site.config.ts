// src/site.config.ts

export const siteConfig = {
  author: 'Xiaoshuu',
  title: 'Xiaoshuu\'s Blog',
  description: 'A student',
  social: {
    github: 'https://github.com/xiaoshuu',
    // twitter: '...',
  }
};

export const statusConfig = [
  {
    label: '间歇性努力选手',
    type: 'focus',   // 绿色圆点
  },
  // {
  //   label: 'Debugging Drones',
  //   type: 'warning', // 橙色圆点
  // },
  // {
  //   label: 'Learning Rust',
  //   type: 'neutral', // 灰色圆点 (可选)
  // }
];

// 如果你想手动指定右侧标签云的内容，可以在这里写
// 如果留空，我会保留自动从文章里抓取标签的逻辑
export const manualTags = [];

export const techStack = [
  'Ubuntu', 'C++', 'Python'
];