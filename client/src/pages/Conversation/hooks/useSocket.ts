import { useState, useEffect, useCallback, useRef } from "react";
import { WSMessage } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
  createSocket,
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
  // Alternative transport with the WebSocket surface (RtcSocket); default: new WebSocket(uri).
  createSocket?: () => WebSocket;
}) => {
  const lastMessageTime = useRef<null|number>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isWaitingForHandshake, setIsWaitingForHandshake] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socket || !isConnected) {
        return;
      }
      socket.send(encodeMessage(message));
    },
    [isConnected],
  );

  const onConnect = useCallback(() => {
    setIsWaitingForHandshake(true);
  }, [setIsWaitingForHandshake]);

  const onDisconnect = useCallback(() => {
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    setIsConnected(false);
    setIsWaitingForHandshake(false);
  }, [onDisconnectProp]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      lastMessageTime.current = Date.now();
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type == "handshake") {
        setIsConnected(true);
        setIsWaitingForHandshake(false);
      }
      if (!onMessage) {
        return;
      }
      onMessage(message);
    },
    [onMessage, setIsConnected],
  );

  const start = useCallback(() => {
    const ws = createSocket ? createSocket() : new WebSocket(uri);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", onDisconnect);
    ws.addEventListener("message", onMessageEvent);
    setSocket(ws);
    lastMessageTime.current = Date.now();
  }, [uri, onMessage, onDisconnectProp, createSocket]);

  const stop = useCallback(() => {
      setIsConnected(false);
      setIsWaitingForHandshake(false);
      if (onDisconnectProp) {
        onDisconnectProp();
      }
      socket?.close();
      setSocket(null);
  }, [socket]);

  useEffect(() => {
    if(!isConnected){
      return;
    }
    let intervalId = setInterval(() => {
      if (lastMessageTime.current && Date.now() - lastMessageTime.current > 10000) {
        socket?.close();
        onDisconnect();
        clearInterval(intervalId);
      }
    }, 500);

    return () => {
      lastMessageTime.current = null;
      clearInterval(intervalId);
    };
  }, [isConnected, socket]);

  return {
    isConnected,
    isWaitingForHandshake,
    socket,
    sendMessage,
    start,
    stop,
  };
};
