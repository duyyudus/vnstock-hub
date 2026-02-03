# VNStock Hub - Frontend

This is the frontend component of the VNStock Hub, a modern web dashboard for tracking the Vietnam stock market. It is built with React, TypeScript, and Vite, featuring a sleek UI powered by TailwindCSS and DaisyUI.

## 🚀 Tech Stack

- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite](https://vitejs.dev/)
- **Language:** [TypeScript](https://www.typescriptlang.org/)
- **Styling:** [TailwindCSS](https://tailwindcss.com/) & [DaisyUI](https://daisyui.com/)
- **Charts:** [Recharts](https://recharts.org/)
- **API Client:** [Axios](https://axios-http.com/)
- **State/Auth:** Custom hooks and context for JWT authentication.

## 📦 Project Structure

```text
frontend/
├── src/
│   ├── api/            # Centralized API client (stockApi.ts)
│   ├── components/     # Shared UI components (Navigation, SyncIndicator)
│   ├── features/       # Feature-based modules
│   │   ├── auth/       # Login/Register widgets and user hooks
│   │   ├── dashboard/  # Main hub: Indices, Funds, Portfolio tabs
│   │   │   ├── banner/ # Index market banners
│   │   │   ├── funds/  # Fund analysis charts and selectors
│   │   │   ├── indices/# Stock tables, growth charts, industry filters
│   │   │   └── portfolio/ # Portfolio tracking and management
│   ├── App.tsx         # Root layout and routing
│   └── main.tsx        # Application entry point
├── tailwind.config.js  # Tailwind CSS & DaisyUI configuration
└── package.json        # Project dependencies and scripts
```

## 🛠️ Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file:
   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   ```

### Running the Application

Start the development server:
```bash
npm run dev
```
Access at: `http://localhost:5173`

## ✨ Key Features

- **Indices Dashboard:** Dynamic index selection (VN30, VN100, etc.), industry-based filtering, and stock bookmarking.
- **Mutual Fund Explorer:** Visual analysis of fund NAV performance, risk-return profiles, and detailed holding allocations.
- **Portfolio Tracking:** Manage your stock positions and view real-time performance summaries.
- **Interactive Visualizations:** Responsive growth charts, volume history, and financial performance popups.
- **Secure Authentication:** User registration and login to persist bookmarks and portfolio data.

## 🎨 Design System

- **Modern UI:** Clean, glassmorphic design using DaisyUI components.
- **Dark Mode:** Fully optimized for professional dark theme environments.
- **Responsive:** Mobile-friendly layouts for market tracking on the go.