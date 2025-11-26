/**
 * MUSE Vision - Internationalization System
 * Supports: English, Korean, Japanese, Chinese
 */

const I18N = {
    currentLang: 'en',

    translations: {
        en: {
            nav: {
                features: 'Features',
                demo: 'Live Demo',
                models: 'Models'
            },
            hero: {
                badge: 'Computer Vision AI',
                title1: 'See the World',
                title2: 'Through AI Eyes',
                subtitle: 'Enterprise-grade computer vision platform with real-time object detection, face recognition, and semantic image search powered by state-of-the-art deep learning.',
                stat1: 'Face Recognition Accuracy',
                stat2: 'Detection Speed',
                stat3: 'Object Categories',
                stat4: 'Images Indexed',
                tryDemo: 'Try Live Demo',
                viewGithub: 'View on GitHub'
            },
            features: {
                title: 'Powerful Vision Capabilities',
                subtitle: 'State-of-the-art deep learning models for every computer vision task',
                detection: {
                    title: 'Object Detection',
                    desc: 'YOLOv8-powered real-time detection with 100+ object categories and custom model training support.',
                    item1: 'Real-time Detection (12ms)',
                    item2: '100+ Categories',
                    item3: 'Custom Training',
                    item4: 'Video Stream Analysis'
                },
                face: {
                    title: 'Face Recognition',
                    desc: 'Industry-leading 99.8% accuracy with ArcFace, anti-spoofing liveness detection, and automatic clustering.',
                    item1: '99.8% Accuracy (ArcFace)',
                    item2: 'Liveness Detection',
                    item3: 'Auto Clustering',
                    item4: 'RetinaFace Detection'
                },
                search: {
                    title: 'Image Search',
                    desc: 'CLIP-powered semantic search enabling both visual similarity and text-to-image queries at scale.',
                    item1: 'Visual Similarity Search',
                    item2: 'Text-to-Image (CLIP)',
                    item3: '10M+ Image Index',
                    item4: 'Real-time Indexing'
                },
                cctv: {
                    title: 'CCTV Integration',
                    desc: 'Live RTSP/HLS streaming with motion detection, person tracking, and real-time event alerts.',
                    item1: 'RTSP/HLS Streaming',
                    item2: 'Motion Detection',
                    item3: 'Multi-Camera Tracking',
                    item4: 'Real-time Alerts'
                }
            },
            demo: {
                title: 'Interactive Demo',
                subtitle: "Experience MUSE Vision's capabilities in real-time",
                tabs: {
                    detection: 'Object Detection',
                    face: 'Face Recognition',
                    search: 'Image Search'
                },
                detection: {
                    upload: 'Drag & drop an image or click to upload',
                    hint: 'Supports JPG, PNG, WEBP',
                    results: 'Detection Results',
                    empty: 'Upload an image to detect objects',
                    try: 'Try samples:'
                },
                face: {
                    upload: 'Upload an image with faces',
                    hint: "We'll detect and analyze all faces",
                    results: 'Face Analysis',
                    empty: 'Upload an image to detect faces',
                    try: 'Try samples:'
                },
                search: {
                    textMode: 'Text Search',
                    imageMode: 'Image Search',
                    placeholder: 'a dog playing in the park...',
                    uploadImage: 'Upload query image',
                    empty: 'Search for images using text or upload a query image',
                    suggestions: 'Try:'
                }
            },
            models: {
                title: 'AI Models & Performance',
                subtitle: 'Powered by state-of-the-art deep learning architectures',
                table: {
                    task: 'Task',
                    model: 'Model',
                    accuracy: 'Accuracy',
                    speed: 'Speed (GPU)'
                }
            },
            arch: {
                title: 'System Architecture',
                subtitle: 'Scalable, production-ready infrastructure',
                sources: 'Input Sources',
                services: 'AI Services',
                storage: 'Storage Layer'
            },
            tech: {
                title: 'Technology Stack'
            },
            cta: {
                title: 'Ready to Build with MUSE Vision?',
                subtitle: 'Get started with our open-source platform today',
                github: 'View on GitHub',
                docs: 'Quick Start Guide'
            }
        },
        ko: {
            nav: {
                features: '기능',
                demo: '라이브 데모',
                models: '모델'
            },
            hero: {
                badge: '컴퓨터 비전 AI',
                title1: '세상을 바라보다',
                title2: 'AI의 눈으로',
                subtitle: '최첨단 딥러닝 기반의 실시간 객체 탐지, 얼굴 인식, 시맨틱 이미지 검색을 제공하는 엔터프라이즈급 컴퓨터 비전 플랫폼입니다.',
                stat1: '얼굴 인식 정확도',
                stat2: '탐지 속도',
                stat3: '객체 카테고리',
                stat4: '인덱싱된 이미지',
                tryDemo: '라이브 데모 체험',
                viewGithub: 'GitHub에서 보기'
            },
            features: {
                title: '강력한 비전 기능',
                subtitle: '모든 컴퓨터 비전 작업을 위한 최첨단 딥러닝 모델',
                detection: {
                    title: '객체 탐지',
                    desc: 'YOLOv8 기반 실시간 탐지, 100+ 객체 카테고리 및 커스텀 모델 학습 지원.',
                    item1: '실시간 탐지 (12ms)',
                    item2: '100+ 카테고리',
                    item3: '커스텀 학습',
                    item4: '비디오 스트림 분석'
                },
                face: {
                    title: '얼굴 인식',
                    desc: 'ArcFace 기반 99.8% 정확도, 스푸핑 방지 생체 인식, 자동 클러스터링.',
                    item1: '99.8% 정확도 (ArcFace)',
                    item2: '생체 인식',
                    item3: '자동 클러스터링',
                    item4: 'RetinaFace 탐지'
                },
                search: {
                    title: '이미지 검색',
                    desc: 'CLIP 기반 시맨틱 검색으로 시각적 유사도 및 텍스트-이미지 쿼리 대규모 지원.',
                    item1: '시각적 유사도 검색',
                    item2: '텍스트-이미지 (CLIP)',
                    item3: '1000만+ 이미지 인덱스',
                    item4: '실시간 인덱싱'
                },
                cctv: {
                    title: 'CCTV 통합',
                    desc: 'RTSP/HLS 라이브 스트리밍, 모션 감지, 인물 추적, 실시간 이벤트 알림.',
                    item1: 'RTSP/HLS 스트리밍',
                    item2: '모션 감지',
                    item3: '다중 카메라 추적',
                    item4: '실시간 알림'
                }
            },
            demo: {
                title: '인터랙티브 데모',
                subtitle: 'MUSE Vision의 기능을 실시간으로 체험하세요',
                tabs: {
                    detection: '객체 탐지',
                    face: '얼굴 인식',
                    search: '이미지 검색'
                },
                detection: {
                    upload: '이미지를 드래그하거나 클릭하여 업로드',
                    hint: 'JPG, PNG, WEBP 지원',
                    results: '탐지 결과',
                    empty: '객체 탐지를 위해 이미지를 업로드하세요',
                    try: '샘플 시도:'
                },
                face: {
                    upload: '얼굴이 있는 이미지를 업로드하세요',
                    hint: '모든 얼굴을 감지하고 분석합니다',
                    results: '얼굴 분석',
                    empty: '얼굴 감지를 위해 이미지를 업로드하세요',
                    try: '샘플 시도:'
                },
                search: {
                    textMode: '텍스트 검색',
                    imageMode: '이미지 검색',
                    placeholder: '공원에서 노는 강아지...',
                    uploadImage: '쿼리 이미지 업로드',
                    empty: '텍스트로 검색하거나 쿼리 이미지를 업로드하세요',
                    suggestions: '시도해보세요:'
                }
            },
            models: {
                title: 'AI 모델 & 성능',
                subtitle: '최첨단 딥러닝 아키텍처로 구동',
                table: {
                    task: '작업',
                    model: '모델',
                    accuracy: '정확도',
                    speed: '속도 (GPU)'
                }
            },
            arch: {
                title: '시스템 아키텍처',
                subtitle: '확장 가능한 프로덕션 레디 인프라',
                sources: '입력 소스',
                services: 'AI 서비스',
                storage: '스토리지 레이어'
            },
            tech: {
                title: '기술 스택'
            },
            cta: {
                title: 'MUSE Vision으로 시작할 준비가 되셨나요?',
                subtitle: '오픈소스 플랫폼으로 오늘 시작하세요',
                github: 'GitHub에서 보기',
                docs: '빠른 시작 가이드'
            }
        },
        ja: {
            nav: {
                features: '機能',
                demo: 'ライブデモ',
                models: 'モデル'
            },
            hero: {
                badge: 'コンピュータビジョンAI',
                title1: '世界を見る',
                title2: 'AIの目で',
                subtitle: '最先端のディープラーニングによるリアルタイム物体検出、顔認識、セマンティック画像検索を提供するエンタープライズグレードのコンピュータビジョンプラットフォーム。',
                stat1: '顔認識精度',
                stat2: '検出速度',
                stat3: 'オブジェクトカテゴリ',
                stat4: 'インデックス画像数',
                tryDemo: 'ライブデモを試す',
                viewGithub: 'GitHubで見る'
            },
            features: {
                title: '強力なビジョン機能',
                subtitle: 'すべてのコンピュータビジョンタスクに対応する最先端のディープラーニングモデル',
                detection: {
                    title: '物体検出',
                    desc: 'YOLOv8によるリアルタイム検出、100以上のカテゴリ、カスタムモデル学習をサポート。',
                    item1: 'リアルタイム検出 (12ms)',
                    item2: '100以上のカテゴリ',
                    item3: 'カスタム学習',
                    item4: 'ビデオストリーム分析'
                },
                face: {
                    title: '顔認識',
                    desc: 'ArcFaceによる99.8%の精度、なりすまし防止の生体認証、自動クラスタリング。',
                    item1: '99.8%精度 (ArcFace)',
                    item2: '生体認証',
                    item3: '自動クラスタリング',
                    item4: 'RetinaFace検出'
                },
                search: {
                    title: '画像検索',
                    desc: 'CLIPによるセマンティック検索で、視覚的類似性とテキスト-画像クエリを大規模にサポート。',
                    item1: '視覚的類似性検索',
                    item2: 'テキスト-画像 (CLIP)',
                    item3: '1000万以上の画像インデックス',
                    item4: 'リアルタイムインデックス'
                },
                cctv: {
                    title: 'CCTV統合',
                    desc: 'RTSP/HLSライブストリーミング、動き検出、人物追跡、リアルタイムイベントアラート。',
                    item1: 'RTSP/HLSストリーミング',
                    item2: '動き検出',
                    item3: 'マルチカメラ追跡',
                    item4: 'リアルタイムアラート'
                }
            },
            demo: {
                title: 'インタラクティブデモ',
                subtitle: 'MUSE Visionの機能をリアルタイムで体験',
                tabs: {
                    detection: '物体検出',
                    face: '顔認識',
                    search: '画像検索'
                },
                detection: {
                    upload: '画像をドラッグまたはクリックしてアップロード',
                    hint: 'JPG、PNG、WEBPをサポート',
                    results: '検出結果',
                    empty: '物体検出のために画像をアップロードしてください',
                    try: 'サンプルを試す:'
                },
                face: {
                    upload: '顔のある画像をアップロード',
                    hint: 'すべての顔を検出・分析します',
                    results: '顔分析',
                    empty: '顔検出のために画像をアップロードしてください',
                    try: 'サンプルを試す:'
                },
                search: {
                    textMode: 'テキスト検索',
                    imageMode: '画像検索',
                    placeholder: '公園で遊ぶ犬...',
                    uploadImage: 'クエリ画像をアップロード',
                    empty: 'テキストで検索するか、クエリ画像をアップロード',
                    suggestions: '試してみる:'
                }
            },
            models: {
                title: 'AIモデルとパフォーマンス',
                subtitle: '最先端のディープラーニングアーキテクチャで駆動',
                table: {
                    task: 'タスク',
                    model: 'モデル',
                    accuracy: '精度',
                    speed: '速度 (GPU)'
                }
            },
            arch: {
                title: 'システムアーキテクチャ',
                subtitle: 'スケーラブルな本番環境対応インフラ',
                sources: '入力ソース',
                services: 'AIサービス',
                storage: 'ストレージレイヤー'
            },
            tech: {
                title: '技術スタック'
            },
            cta: {
                title: 'MUSE Visionで構築する準備はできましたか？',
                subtitle: 'オープンソースプラットフォームで今すぐ始めましょう',
                github: 'GitHubで見る',
                docs: 'クイックスタートガイド'
            }
        },
        zh: {
            nav: {
                features: '功能',
                demo: '在线演示',
                models: '模型'
            },
            hero: {
                badge: '计算机视觉AI',
                title1: '通过AI的眼睛',
                title2: '观察世界',
                subtitle: '企业级计算机视觉平台，提供由最先进深度学习驱动的实时目标检测、人脸识别和语义图像搜索。',
                stat1: '人脸识别准确率',
                stat2: '检测速度',
                stat3: '目标类别',
                stat4: '已索引图像',
                tryDemo: '体验在线演示',
                viewGithub: '在GitHub上查看'
            },
            features: {
                title: '强大的视觉能力',
                subtitle: '为每个计算机视觉任务提供最先进的深度学习模型',
                detection: {
                    title: '目标检测',
                    desc: '基于YOLOv8的实时检测，100+目标类别，支持自定义模型训练。',
                    item1: '实时检测 (12ms)',
                    item2: '100+类别',
                    item3: '自定义训练',
                    item4: '视频流分析'
                },
                face: {
                    title: '人脸识别',
                    desc: '基于ArcFace的99.8%准确率，防欺骗活体检测，自动聚类。',
                    item1: '99.8%准确率 (ArcFace)',
                    item2: '活体检测',
                    item3: '自动聚类',
                    item4: 'RetinaFace检测'
                },
                search: {
                    title: '图像搜索',
                    desc: '基于CLIP的语义搜索，支持大规模视觉相似性和文本到图像查询。',
                    item1: '视觉相似性搜索',
                    item2: '文本到图像 (CLIP)',
                    item3: '1000万+图像索引',
                    item4: '实时索引'
                },
                cctv: {
                    title: 'CCTV集成',
                    desc: 'RTSP/HLS直播流，运动检测，人员追踪，实时事件警报。',
                    item1: 'RTSP/HLS流媒体',
                    item2: '运动检测',
                    item3: '多摄像头追踪',
                    item4: '实时警报'
                }
            },
            demo: {
                title: '交互式演示',
                subtitle: '实时体验MUSE Vision的功能',
                tabs: {
                    detection: '目标检测',
                    face: '人脸识别',
                    search: '图像搜索'
                },
                detection: {
                    upload: '拖放图像或点击上传',
                    hint: '支持JPG、PNG、WEBP',
                    results: '检测结果',
                    empty: '上传图像以检测目标',
                    try: '尝试示例:'
                },
                face: {
                    upload: '上传包含人脸的图像',
                    hint: '我们将检测并分析所有人脸',
                    results: '人脸分析',
                    empty: '上传图像以检测人脸',
                    try: '尝试示例:'
                },
                search: {
                    textMode: '文本搜索',
                    imageMode: '图像搜索',
                    placeholder: '在公园玩耍的狗...',
                    uploadImage: '上传查询图像',
                    empty: '使用文本搜索或上传查询图像',
                    suggestions: '尝试:'
                }
            },
            models: {
                title: 'AI模型与性能',
                subtitle: '由最先进的深度学习架构驱动',
                table: {
                    task: '任务',
                    model: '模型',
                    accuracy: '准确率',
                    speed: '速度 (GPU)'
                }
            },
            arch: {
                title: '系统架构',
                subtitle: '可扩展的生产就绪基础设施',
                sources: '输入源',
                services: 'AI服务',
                storage: '存储层'
            },
            tech: {
                title: '技术栈'
            },
            cta: {
                title: '准备好使用MUSE Vision构建了吗？',
                subtitle: '立即开始使用我们的开源平台',
                github: '在GitHub上查看',
                docs: '快速入门指南'
            }
        }
    },

    init() {
        const saved = localStorage.getItem('muse_vision_lang');
        const browserLang = navigator.language.split('-')[0];
        this.currentLang = saved || (['ko', 'ja', 'zh'].includes(browserLang) ? browserLang : 'en');
        this.apply();
        this.setupSelector();
    },

    t(key) {
        const keys = key.split('.');
        let value = this.translations[this.currentLang];
        for (const k of keys) {
            value = value?.[k];
        }
        return value || key;
    },

    apply() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const text = this.t(key);
            if (text) el.textContent = text;
        });

        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            const text = this.t(key);
            if (text) el.placeholder = text;
        });

        document.documentElement.lang = this.currentLang;
    },

    setLang(lang) {
        if (this.translations[lang]) {
            this.currentLang = lang;
            localStorage.setItem('muse_vision_lang', lang);
            this.apply();
        }
    },

    setupSelector() {
        const container = document.getElementById('lang-selector');
        if (!container) return;

        const select = document.createElement('select');
        select.innerHTML = `
            <option value="en" ${this.currentLang === 'en' ? 'selected' : ''}>English</option>
            <option value="ko" ${this.currentLang === 'ko' ? 'selected' : ''}>한국어</option>
            <option value="ja" ${this.currentLang === 'ja' ? 'selected' : ''}>日本語</option>
            <option value="zh" ${this.currentLang === 'zh' ? 'selected' : ''}>中文</option>
        `;
        select.addEventListener('change', (e) => this.setLang(e.target.value));
        container.appendChild(select);
    }
};

document.addEventListener('DOMContentLoaded', () => I18N.init());
