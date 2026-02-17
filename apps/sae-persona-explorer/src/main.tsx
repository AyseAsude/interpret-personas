import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import PersonaDriftView from "./drift/PersonaDriftView";
import "./styles.css";
import "./drift/styles.css";

type Route = "sae" | "drift";

function routeFromHash(hash: string): Route {
  const normalized = hash.startsWith("#") ? hash.slice(1) : hash;
  if (normalized.startsWith("/persona-drift")) {
    return "drift";
  }
  return "sae";
}

function RoutedApp() {
  const [route, setRoute] = React.useState<Route>(() => {
    if (typeof window === "undefined") {
      return "sae";
    }
    return routeFromHash(window.location.hash);
  });

  React.useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }
    const onHashChange = () => {
      setRoute(routeFromHash(window.location.hash));
    };
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);

  if (route === "drift") {
    return <PersonaDriftView />;
  }

  return <App />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RoutedApp />
  </React.StrictMode>,
);
