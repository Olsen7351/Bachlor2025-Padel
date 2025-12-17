import { auth } from "../../firebase/firebase.ts";

export async function apiFetch(
    input: RequestInfo,
    init: RequestInit = {}
) {
    const token = localStorage.getItem("idToken");

    const res = await fetch(input, {
        ...init,
        headers: {
            ...init.headers,
            Authorization: token ? `Bearer ${token}` : "",
        },
    });

    if (res.status === 401 || res.status === 403) {
        console.warn("Auth expired – redirecting to login");

        await auth.signOut();
        localStorage.removeItem("idToken");

        window.location.href = "/";
        throw new Error("Unauthorized");
    }

    return res;
}