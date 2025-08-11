import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";

const GreetingForm: React.FC = () => {
  const [name, setName] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const resp = await invoke<string>("greet_py", { name });
    setMessage(resp);
  };

  return (
      <>
        <form onSubmit={handleSubmit} className="row">
          <input
              id="greet-input"
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
              placeholder="Enter a name..."
          />
          <button type="submit">Greet</button>
        </form>
        {message && <p>{message}</p>}
      </>
  );
};

export default GreetingForm;