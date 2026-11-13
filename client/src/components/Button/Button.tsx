import { FC } from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  // A chosen option.
  selected?: boolean;
  // The main action.
  accent?: boolean;
  large?: boolean;
};

export const capsuleClass = ({ selected, accent, large }: { selected?: boolean; accent?: boolean; large?: boolean }) =>
  `inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-full transition-opacity active:opacity-75 disabled:opacity-40 disabled:cursor-not-allowed ${
    large ? "px-[22px] py-[11px] text-[15px] font-semibold" : "px-3.5 py-1.5 text-[13px] font-medium"
  } ${selected || accent ? "bg-accent text-white" : "bg-fill text-ink"}`;

// Haru's capsule button: red-orange and filled for the main action or a chosen option, grey otherwise.
export const Button: FC<ButtonProps> = ({ children, className, selected, accent, large, ...props }) => {
  return (
    <button type="button" className={`${capsuleClass({ selected, accent, large })} ${className ?? ""}`} {...props}>
      {children}
    </button>
  );
};

export const SwitchButton: FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({ children, className, ...props }) => {
  return (
    <button type="button" className={`text-[13px] text-accent hover:underline disabled:opacity-40 ${className ?? ""}`} {...props}>
      {children}
    </button>
  );
};
