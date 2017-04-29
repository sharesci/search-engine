export class User {
    username: string = '';
    password: string = '';
    firstname: string = '';
    lastname: string = '';
    institution: string = '';
    self_bio: string = '';

    constructor() { }
}

export interface IUserWrapper {
    errno: number,
    errstr: string,
    userJson: User
}