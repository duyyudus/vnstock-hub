import { useEffect, useState } from 'react';
import { authStorage, AUTH_EVENT } from '../../api/stockApi';
import type { AuthUser } from '../../api/stockApi';

export const useAuthUser = () => {
    const [user, setUser] = useState<AuthUser | null>(() => authStorage.getUser());

    useEffect(() => {
        const handleChange = () => {
            setUser(authStorage.getUser());
        };

        window.addEventListener(AUTH_EVENT, handleChange);
        window.addEventListener('storage', handleChange);

        return () => {
            window.removeEventListener(AUTH_EVENT, handleChange);
            window.removeEventListener('storage', handleChange);
        };
    }, []);

    return user;
};

export default useAuthUser;
