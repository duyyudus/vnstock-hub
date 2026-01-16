# VNStock Hub - Frontend

This is the frontend component of the VNStock Hub, a modern web dashboard for tracking the Vietnam stock market. It is built with React, TypeScript, and Vite, featuring a sleek UI powered by TailwindCSS and DaisyUI.

## 🚀 Tech Stack

- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite](https://vitejs.dev/)
- **Language:** [TypeScript](https://www.typescriptlang.org/)
- **Styling:** [TailwindCSS](https://tailwindcss.com/) & [DaisyUI](https://daisyui.com/)
- **API Client:** [Axios](https://axios-http.com/)
- **Icons:** [Lucide React](https://lucide.dev/) (implied/used in components)

## 📦 Project Structure

```text
frontend/
├── src/
│   ├── api/            # API client and service definitions
│   ├── components/     # Reusable UI components (Navigation, Layouts)
│   ├── features/       # Feature-based modules (Dashboard, Stocks)
│   ├── assets/         # Static assets (images, fonts)
│   ├── App.tsx         # Root component
│   ├── main.tsx        # Application entry point
│   └── index.css       # Global styles and Tailwind directives
├── public/             # Static public assets
├── tailwind.config.js  # Tailwind CSS configuration
└── package.json        # Project dependencies and scripts
```

## 🛠️ Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create a `.env` file with the following variable:

   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   ```

### Running the Application

Start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`.

## ✨ Key Features

- **VN-100 Table:** A real-time dashboard displaying the top 100 stocks in the Vietnam market.
- **Dynamic Navigation:** A clean tab-based navigation system to switch between different market views.
- **Premium Design:** Modern, responsive UI with dark mode support and glassmorphism elements.
- **Data Formatting:** Automatically formats market capitalization and ticker symbols for clarity.

## 🎨 Design System

The project uses a custom design system based on:

- **Primary Palette:** Professional blues and greens for financial data.
- **Typography:** Modern sans-serif fonts (Inter/system-default).
- **Responsive Layout:** Mobile-first approach ensuring accessibility across all devices.
