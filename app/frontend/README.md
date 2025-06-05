# SAGED Frontend

Modern React TypeScript frontend providing an intuitive interface for bias analysis and visualization in the SAGED platform.

![React](https://img.shields.io/badge/React-18+-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue) ![Vite](https://img.shields.io/badge/Vite-5+-purple) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3+-blue)

## 🚀 Quick Start

### Development Server

```bash
# From project root
cd app/frontend
npm install
npm run dev
```

### Production Build

```bash
npm run build
npm run preview  # Preview production build
```

### Access Points

- **Web App**: http://localhost:3000
- **Development**: Hot reload enabled
- **Preview**: http://localhost:4173 (after build)

## 🏗️ Architecture

```
frontend/src/
├── components/          # Reusable UI components
│   ├── ui/             # shadcn/ui components
│   │   ├── button.tsx  # Button component
│   │   ├── dialog.tsx  # Modal dialogs
│   │   ├── input.tsx   # Form inputs
│   │   ├── select.tsx  # Dropdown selects
│   │   └── ...         # Other UI primitives
│   └── Layout.tsx      # Main application layout
├── pages/              # Application screens/routes
│   ├── Dashboard.tsx   # Overview and metrics
│   ├── Benchmarks.tsx  # Benchmark management
│   ├── Experiments.tsx # Experiment execution
│   └── Analysis.tsx    # Results visualization
├── lib/                # Utility functions
│   └── utils.ts        # Common utilities (cn, etc.)
├── App.tsx             # Main application component
├── main.tsx           # Application entry point
└── index.css          # Global styles & Tailwind
```

## 🎨 UI Components

### Design System

Built with **shadcn/ui** components providing:

- **Consistent Design**: Unified component library
- **Accessibility**: ARIA-compliant components
- **Customizable**: Tailwind CSS styling
- **Type Safety**: Full TypeScript support

### Core Components

```typescript
// Button variants
<Button variant="default | destructive | outline | secondary | ghost | link">
  Action
</Button>

// Modal dialogs
<Dialog>
  <DialogTrigger asChild>
    <Button>Open Dialog</Button>
  </DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
    </DialogHeader>
    Content here
  </DialogContent>
</Dialog>

// Form inputs
<Input
  type="text"
  placeholder="Enter text..."
  value={value}
  onChange={handleChange}
/>

// Select dropdowns
<Select onValueChange={setValue}>
  <SelectTrigger>
    <SelectValue placeholder="Select option" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="option1">Option 1</SelectItem>
  </SelectContent>
</Select>
```

## 📱 Pages & Features

### 1. Dashboard (`pages/Dashboard.tsx`)

**Overview and metrics dashboard**

- 📊 **Platform Statistics**: Benchmarks, experiments, results summary
- 📈 **Recent Activity**: Latest experiments and their status
- 🎯 **Quick Actions**: Create benchmark, run experiment shortcuts
- 📋 **System Status**: Backend health, database connectivity

### 2. Benchmarks (`pages/Benchmarks.tsx`)

**Bias test configuration and management**

- ➕ **Create Benchmarks**: Multi-step form with validation
- 📝 **Configuration Options**:
  - Domain selection (employment, healthcare, education)
  - Bias categories (gender, race, nationality, religion, profession, age)
  - Data tier selection (lite, keywords, questions, etc.)
- 📋 **Benchmark List**: Searchable, sortable table
- ⚙️ **Management**: Edit, delete, duplicate benchmarks
- 📊 **Status Tracking**: Draft, ready, processing states

### 3. Experiments (`pages/Experiments.tsx`)

**Bias analysis execution and monitoring**

- 🚀 **Launch Experiments**: Select benchmark and configure parameters
- 📊 **Real-time Progress**: Progress bars and status updates
- ⏱️ **Execution Monitoring**: Duration, samples processed, current stage
- 📋 **Experiment History**: Past runs and their outcomes
- 🔄 **Restart/Stop**: Control experiment execution

### 4. Analysis (`pages/Analysis.tsx`)

**Results visualization and interpretation**

- 📈 **Charts & Graphs**: Interactive bias metric visualizations
- 📊 **Statistical Summary**: P-values, effect sizes, confidence intervals
- 🔍 **Detailed Results**: Feature-level bias analysis
- 📋 **Comparison Tools**: Compare across experiments
- 📥 **Export Options**: Download results in various formats

## 🔧 State Management

### TanStack Query

Handles server state and caching:

```typescript
// Fetch benchmarks with caching
const {
  data: benchmarks,
  isLoading,
  error,
} = useQuery({
  queryKey: ["benchmarks"],
  queryFn: fetchBenchmarks,
  staleTime: 5 * 60 * 1000, // 5 minutes
  gcTime: 10 * 60 * 1000, // 10 minutes
});

// Create benchmark mutation
const createMutation = useMutation({
  mutationFn: createBenchmark,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
    toast({ title: "Benchmark created successfully" });
  },
  onError: () => {
    toast({ title: "Error", variant: "destructive" });
  },
});
```

### Local State

Using React hooks for component state:

```typescript
// Form state management
const [formData, setFormData] = useState<BenchmarkFormData>({
  name: "",
  domain: "",
  categories: [],
  data_tier: "lite",
});

// Modal state
const [isOpen, setIsOpen] = useState(false);
```

## 🎨 Styling

### Tailwind CSS

Utility-first CSS framework:

```typescript
// Responsive design
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

// Component styling
<Button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">

// Conditional styles
<Badge className={cn(
  "px-2 py-1 rounded-full text-xs",
  status === "ready" ? "bg-green-100 text-green-800" : "bg-gray-100"
)}>
```

### CSS Variables

Theme customization via CSS variables:

```css
/* index.css */
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 222.2 47.4% 11.2%;
  --primary-foreground: 210 40% 98%;
  --secondary: 210 40% 96%;
  --accent: 210 40% 96%;
  --destructive: 0 84.2% 60.2%;
}
```

## 🔄 API Integration

### REST Client

Fetch-based API communication:

```typescript
// API helper functions
const fetchBenchmarks = async (): Promise<{ benchmarks: Benchmark[] }> => {
  const response = await fetch("/api/v1/benchmarks/");
  if (!response.ok) throw new Error("Failed to fetch benchmarks");
  return response.json();
};

const createBenchmark = async (
  data: CreateBenchmarkData
): Promise<Benchmark> => {
  const response = await fetch("/api/v1/benchmarks/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error("Failed to create benchmark");
  return response.json();
};
```

### Error Handling

Comprehensive error management:

```typescript
// Global error boundary
const { toast } = useToast();

// Mutation error handling
const mutation = useMutation({
  mutationFn: createBenchmark,
  onError: (error: Error) => {
    toast({
      title: "Error",
      description: error.message,
      variant: "destructive",
    });
  },
});
```

## 🧪 Development

### Adding New Pages

1. **Create Component** in `pages/`
2. **Add Route** in `App.tsx`
3. **Update Navigation** in `Layout.tsx`

```typescript
// pages/NewPage.tsx
import React from "react";

const NewPage: React.FC = () => {
  return (
    <div>
      <h1>New Page</h1>
    </div>
  );
};

export default NewPage;

// App.tsx
import NewPage from "./pages/NewPage";

<Routes>
  <Route path="/new-page" element={<NewPage />} />
</Routes>;
```

### Adding Components

1. **Create in `components/`** or `components/ui/`
2. **Follow shadcn patterns** for UI components
3. **Add TypeScript interfaces**

```typescript
// components/CustomComponent.tsx
import React from "react";
import { cn } from "@/lib/utils";

interface CustomComponentProps {
  title: string;
  variant?: "default" | "highlight";
  className?: string;
}

const CustomComponent: React.FC<CustomComponentProps> = ({
  title,
  variant = "default",
  className,
}) => {
  return (
    <div
      className={cn(
        "p-4 rounded",
        variant === "highlight" ? "bg-yellow-100" : "bg-gray-100",
        className
      )}
    >
      <h3>{title}</h3>
    </div>
  );
};

export default CustomComponent;
```

### Installing New UI Components

```bash
# Install shadcn/ui components
npx shadcn@latest add card
npx shadcn@latest add table
npx shadcn@latest add chart

# Custom installation
npm install lucide-react    # Icons
npm install recharts        # Charts
npm install @tanstack/react-query  # State management
```

## 🧪 Testing

### Component Testing

```bash
npm run test              # Run all tests
npm run test:watch       # Watch mode
npm run test:coverage    # Coverage report
```

### Manual Testing

```bash
# Development testing
npm run dev

# Production testing
npm run build && npm run preview

# Accessibility testing
npm run test:a11y
```

### Test Examples

```typescript
// Test utilities
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Benchmarks from "../pages/Benchmarks";

// Component test
test("renders benchmark creation form", () => {
  const queryClient = new QueryClient();

  render(
    <QueryClientProvider client={queryClient}>
      <Benchmarks />
    </QueryClientProvider>
  );

  expect(screen.getByText("Create Benchmark")).toBeInTheDocument();
});
```

## 📱 Responsive Design

### Breakpoints

```css
/* Tailwind breakpoints */
sm: 640px    /* Mobile landscape */
md: 768px    /* Tablet */
lg: 1024px   /* Desktop */
xl: 1280px   /* Large desktop */
```

### Mobile-First Design

```typescript
// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

// Mobile navigation
<div className="block md:hidden">
  <MobileMenu />
</div>

// Responsive text
<h1 className="text-2xl md:text-3xl lg:text-4xl font-bold">
```

## 🚀 Performance

### Code Splitting

```typescript
// Lazy loading pages
import { lazy, Suspense } from "react";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Analysis = lazy(() => import("./pages/Analysis"));

// In routes
<Suspense fallback={<div>Loading...</div>}>
  <Dashboard />
</Suspense>;
```

### Optimization

```typescript
// Memoization
import { memo, useMemo, useCallback } from "react";

const ExpensiveComponent = memo(({ data }) => {
  const processedData = useMemo(() => {
    return data.map((item) => processItem(item));
  }, [data]);

  const handleClick = useCallback(() => {
    // Handler logic
  }, []);

  return <div>{/* Component JSX */}</div>;
});
```

## 🔧 Configuration

### Vite Config

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

### TypeScript Config

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

## 🚀 Deployment

### Build for Production

```bash
npm run build          # Creates dist/ folder
npm run preview        # Test production build
```

### Static Hosting

```bash
# Deploy to Netlify, Vercel, or GitHub Pages
npm run build

# Upload dist/ folder to hosting provider
# Configure routing for SPA (Single Page Application)
```

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🔗 Integration

- **Backend API**: Communicates with FastAPI backend via REST
- **Real-time Updates**: Polling for experiment progress
- **File Uploads**: Benchmark data and configuration files
- **Authentication**: Ready for auth integration (Auth0, Firebase)

---

For more information, see the [main README](../../README.md) or the [backend documentation](../backend/README.md).
