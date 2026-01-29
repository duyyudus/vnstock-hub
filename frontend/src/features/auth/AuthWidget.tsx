import React, { useEffect, useRef, useState } from 'react';
import { authStorage, stockApi } from '../../api/stockApi';
import type { AuthUser } from '../../api/stockApi';

type AuthMode = 'login' | 'register';

const getErrorMessage = (error: unknown) => {
    if (typeof error === 'object' && error && 'response' in error) {
        const response = (error as { response?: { data?: { detail?: string } } }).response;
        if (response?.data?.detail) {
            return response.data.detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Unable to complete the request.';
};

export const AuthWidget: React.FC = () => {
    const dialogRef = useRef<HTMLDialogElement>(null);
    const [mode, setMode] = useState<AuthMode>('login');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [user, setUser] = useState<AuthUser | null>(null);

    useEffect(() => {
        setUser(authStorage.getUser());
    }, []);

    const openModal = (nextMode: AuthMode) => {
        setMode(nextMode);
        setError('');
        setPassword('');
        setConfirmPassword('');
        dialogRef.current?.showModal();
    };

    const closeModal = () => {
        dialogRef.current?.close();
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');

        if (mode === 'register' && password !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        setLoading(true);
        try {
            const response = mode === 'login'
                ? await stockApi.login({ email, password })
                : await stockApi.register({ email, password });
            authStorage.setToken(response.access_token);
            authStorage.setUser(response.user);
            setUser(response.user);
            closeModal();
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    const logout = () => {
        authStorage.clearAll();
        setUser(null);
    };

    return (
        <div className="flex items-center gap-2">
            {user ? (
                <>
                    <span className="hidden sm:inline text-sm text-base-content/70">{user.email}</span>
                    <button className="btn btn-outline btn-sm" onClick={logout}>
                        Log out
                    </button>
                </>
            ) : (
                <>
                    <button className="btn btn-ghost btn-sm" onClick={() => openModal('login')}>
                        Sign in
                    </button>
                    <button className="btn btn-primary btn-sm" onClick={() => openModal('register')}>
                        Create account
                    </button>
                </>
            )}

            <dialog ref={dialogRef} className="modal">
                <div className="modal-box">
                    <h3 className="font-bold text-lg">
                        {mode === 'login' ? 'Welcome back' : 'Create your account'}
                    </h3>
                    <p className="text-sm text-base-content/70 mt-1">
                        Guest mode still works. Sign in to unlock upcoming features.
                    </p>

                    <div className="tabs tabs-boxed mt-4">
                        <button
                            type="button"
                            className={`tab ${mode === 'login' ? 'tab-active' : ''}`}
                            onClick={() => setMode('login')}
                        >
                            Sign in
                        </button>
                        <button
                            type="button"
                            className={`tab ${mode === 'register' ? 'tab-active' : ''}`}
                            onClick={() => setMode('register')}
                        >
                            Register
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="mt-4 space-y-4">
                        <label className="form-control w-full">
                            <div className="label">
                                <span className="label-text">Email</span>
                            </div>
                            <input
                                type="email"
                                required
                                value={email}
                                onChange={(event) => setEmail(event.target.value)}
                                className="input input-bordered w-full"
                                placeholder="you@example.com"
                            />
                        </label>

                        <label className="form-control w-full">
                            <div className="label">
                                <span className="label-text">Password</span>
                                <span className="label-text-alt">8-128 characters</span>
                            </div>
                            <input
                                type="password"
                                required
                                minLength={8}
                                maxLength={128}
                                value={password}
                                onChange={(event) => setPassword(event.target.value)}
                                className="input input-bordered w-full"
                                placeholder="••••••••"
                            />
                        </label>

                        {mode === 'register' ? (
                            <label className="form-control w-full">
                                <div className="label">
                                    <span className="label-text">Confirm password</span>
                                </div>
                                <input
                                    type="password"
                                    required
                                    minLength={8}
                                    maxLength={128}
                                    value={confirmPassword}
                                    onChange={(event) => setConfirmPassword(event.target.value)}
                                    className="input input-bordered w-full"
                                    placeholder="••••••••"
                                />
                            </label>
                        ) : null}

                        {error ? (
                            <div className="alert alert-error text-sm">
                                <span>{error}</span>
                            </div>
                        ) : null}

                        <div className="modal-action">
                            <button type="button" className="btn btn-ghost" onClick={closeModal}>
                                Cancel
                            </button>
                            <button type="submit" className="btn btn-primary" disabled={loading}>
                                {loading ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Create account'}
                            </button>
                        </div>
                    </form>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button>close</button>
                </form>
            </dialog>
        </div>
    );
};
